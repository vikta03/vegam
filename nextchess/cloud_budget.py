"""Conservative quote/benchmark gate. No cloud mutation in this module."""
from dataclasses import dataclass, asdict
import math


@dataclass(frozen=True)
class Budget:
    hourly_services_usd: float
    tax_fraction: float
    total_cap_usd: float | None = 20.
    checkout_fee_usd: float = 0.
    prior_services_usd: float = 0.
    retention_services_usd: float = .75
    reserve_total_usd: float = .50
    max_hours: float = 5.5
    benchmark_services_usd: float = 1.50
    shutdown_seconds: float = 180.
    max_services_usd: float | None = 16.
    minimum_positions_s: float = 50_000.
    minimum_positions_per_dollar: float = 0.

    def __post_init__(self):
        if any(not math.isfinite(v) or v < 0 for v in asdict(self).values() if v is not None):
            raise ValueError('all quote/budget values must be finite and nonnegative')
        if ((self.total_cap_usd is not None and not 0 < self.total_cap_usd <= 20)
                or self.hourly_services_usd <= 0 or self.max_hours <= 0):
            raise ValueError('positive quote and original cap of at most $20, or explicit uncapped mode, required')
        if self.service_ceiling <= self.prior_services_usd+self.retention_services_usd:
            raise ValueError('no rental budget remains')

    @property
    def service_ceiling(self):
        total = math.inf if self.total_cap_usd is None else (self.total_cap_usd-self.checkout_fee_usd-self.reserve_total_usd)/(1+self.tax_fraction)
        return min(math.inf if self.max_services_usd is None else self.max_services_usd, total)

    @property
    def stop_seconds(self):
        available = self.service_ceiling-self.prior_services_usd-self.retention_services_usd
        return min(self.max_hours*3600, available/self.hourly_services_usd*3600)

    @property
    def training_stop_seconds(self):
        return max(0., self.stop_seconds-self.shutdown_seconds)

    @property
    def benchmark_stop_seconds(self):
        return min(self.training_stop_seconds, self.benchmark_services_usd/self.hourly_services_usd*3600)

    def forecast(self, elapsed_seconds, remaining_positions, accepted_positions_s, overhead_seconds=600.):
        values = (elapsed_seconds, remaining_positions, accepted_positions_s, overhead_seconds)
        if any(not math.isfinite(x) or x < 0 for x in values) or accepted_positions_s == 0:
            return dict(approved=False, reason='invalid measured throughput or forecast')
        total_seconds = elapsed_seconds+remaining_positions/accepted_positions_s+overhead_seconds
        positions_per_dollar = accepted_positions_s*3600/(self.hourly_services_usd*(1+self.tax_fraction))
        approved = (accepted_positions_s >= self.minimum_positions_s
                    and positions_per_dollar >= self.minimum_positions_per_dollar
                    and total_seconds <= self.training_stop_seconds)
        services = self.prior_services_usd+self.retention_services_usd+total_seconds/3600*self.hourly_services_usd
        return dict(approved=approved, reason='fits' if approved else 'throughput/deadline gate failed',
                    predicted_seconds=total_seconds, predicted_total_usd=services*(1+self.tax_fraction)+self.checkout_fee_usd,
                    stop_seconds=self.stop_seconds, positions_per_dollar=positions_per_dollar)


def gate(receipt, budget, elapsed_seconds, remaining_positions, *, allow_partial=False):
    """Short synthetic or GPU-only measurements never authorize a full run."""
    for name in ('measured_seconds', 'checkpoint_cycles', 'accepted_positions_s'):
        value = receipt.get(name)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            return dict(approved=False, reason='invalid benchmark evidence')
    if (receipt.get('kind') != 'accepted_end_to_end_training' or receipt.get('measured_seconds', 0) < 300
            or receipt.get('checkpoint_cycles', 0) < 2 or receipt.get('all_ranks_verified') is not True
            or receipt.get('integrity_verified') is not True):
        return dict(approved=False, reason='incomplete benchmark evidence')
    forecast = budget.forecast(elapsed_seconds, remaining_positions, receipt['accepted_positions_s'])
    if not allow_partial:
        return forecast
    # A budget-limited run still requires real, recoverable training evidence.
    # Completion and throughput forecasts remain diagnostics, not admission
    # requirements. Never extend the approved monetary or wall-clock deadline.
    if (not math.isfinite(elapsed_seconds) or elapsed_seconds < 0
            or not math.isfinite(remaining_positions) or remaining_positions < 0
            or receipt['accepted_positions_s'] <= 0):
        return dict(approved=False, reason='invalid partial-run evidence')
    remaining_seconds = max(0., budget.training_stop_seconds-elapsed_seconds-120)
    return dict(approved=remaining_seconds > 0, reason='budget-limited partial training authorized',
                full_target_forecast=forecast, remaining_training_seconds=remaining_seconds,
                estimated_additional_positions=min(remaining_positions,
                    int(remaining_seconds*receipt['accepted_positions_s'])), stop_seconds=budget.stop_seconds)
