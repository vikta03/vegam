#!/usr/bin/env python3
"""Bounded release checks, not a match or strength benchmark."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import selectors
import subprocess
import tempfile
import time

import chess


class UCI:
    def __init__(self, command, provider, cwd, directory):
        self.lines = []
        self.pending = b""
        self.stderr = (directory / "stderr.log").open("w+")
        self.proc = subprocess.Popen([str(command)], cwd=cwd, stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=self.stderr,
                                     env=dict(os.environ, NEXT_PROVIDER=provider))
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.proc.stdout, selectors.EVENT_READ)

    def send(self, line):
        self.proc.stdin.write((line + "\n").encode())
        self.proc.stdin.flush()

    def until(self, prefix, seconds=45):
        end = time.monotonic() + seconds
        result = []
        while time.monotonic() < end:
            while b"\n" in self.pending:
                raw, self.pending = self.pending.split(b"\n", 1)
                line = raw.decode(errors="replace").strip()
                self.lines.append(line)
                result.append(line)
                if line.startswith(prefix):
                    return result
            for key, _ in self.selector.select(min(0.25, max(0, end-time.monotonic()))):
                chunk = os.read(key.fd, 65536)
                if not chunk:
                    raise RuntimeError(f"Engine exited before {prefix}")
                self.pending += chunk
        raise TimeoutError(prefix)

    def close(self):
        try:
            if self.proc.poll() is None:
                self.send("quit")
                self.proc.wait(timeout=5)
            if self.proc.returncode != 0:
                raise RuntimeError(f"Engine exited {self.proc.returncode}")
        finally:
            if self.proc.poll() is None:
                self.proc.kill()
                self.proc.wait()
            self.selector.close()
            self.stderr.close()


def check(package, provider, output, alias=False):
    output.mkdir(parents=True, exist_ok=True)
    models = json.loads((package / "models.json").read_text())
    assert hashlib.sha256((package / "model.onnx").read_bytes()).hexdigest() == models["model.onnx"]["sha256"]
    with tempfile.TemporaryDirectory(prefix="vegam cwd with spaces ") as temporary:
        cwd = Path(temporary)
        command = package / "vegamt-tf3-q"
        if alias:
            link = cwd / "engine alias"
            link.symlink_to(command)
            command = link
        e = UCI(command, provider, cwd, output)
        tests = []
        try:
            e.send("uci")
            options = "\n".join(e.until("uciok"))
            for name, default in [("NextActionValues", "true"), ("NextMovesLeft", "false"),
                                  ("NextOutcomeBlend", "false")]:
                assert f"option name {name} type check default {default}" in options, name
            e.send("isready")
            e.until("readyok")
            e.send("setoption name VerboseMoveStats value true")
            positions = [
                ("start", chess.STARTING_FEN, []),
                ("queen-sacrifice", "1kbr3r/pp6/8/P1n2ppq/2N3n1/4Q1P1/3B1P2/R1R2BK1 w - - 0 1", []),
                ("mate", "7k/5K2/6Q1/8/8/8/8/8 w - - 0 1", []),
                ("promotion", "7k/P7/8/8/8/8/8/6K1 w - - 0 1", []),
            ]
            for name, fen, moves in positions:
                board = chess.Board(fen)
                for move in moves:
                    board.push_uci(move)
                e.send("ucinewgame")
                e.send("position fen " + fen + (" moves " + " ".join(moves) if moves else ""))
                e.send("go nodes 32")
                lines = e.until("bestmove")
                move = chess.Move.from_uci(lines[-1].split()[1])
                assert move in board.legal_moves, name
                if name == "mate":
                    board.push(move)
                    assert board.is_checkmate(), "Immediate mate safety regression"
                tests.append({"name": name, "bestmove": move.uci()})
            # A legal repeated history: expand the only requested child.
            fen = "4r1k1/3p4/8/4P3/8/8/8/4K1N1 b - - 0 1"
            prefix = "d7d5 g1f3 g8h8 f3g1 h8g8 g1f3 g8h8 f3g1"
            e.send("ucinewgame")
            e.send("position fen " + fen + " moves " + prefix)
            e.send("go nodes 20 searchmoves h8g8")
            lines = e.until("bestmove")
            assert lines[-1].split()[1] == "h8g8"
            e.send("isready")
            e.until("readyok")
            e.stderr.flush()
            logs = "\n".join(lines) + (output / "stderr.log").read_text()
            rows = [line for line in logs.splitlines() if "h8g8" in line]
            assert any("(T)" in line and "1.000" in line for line in rows), "Terminal repetition not reported"
            tests.append({"name": "terminal-repetition-child", "bestmove": "h8g8"})
            e.send("ucinewgame")
            e.send("position startpos")
            e.send("go infinite")
            time.sleep(0.25)
            begin = time.monotonic()
            e.send("stop")
            stopped = e.until("bestmove", seconds=5)
            assert chess.Move.from_uci(stopped[-1].split()[1]) in chess.Board().legal_moves
            tests.append({"name": "stop", "seconds": time.monotonic()-begin})
        finally:
            try:
                e.close()
            finally:
                (output / "uci.log").write_text("\n".join(e.lines) + "\n")
        logs = "\n".join(e.lines) + (output / "stderr.log").read_text()
        expected = {"cpu": "CPU", "cuda": "CUDA", "tensorrt": "TensorRT engine"}
        if provider in expected:
            assert "next backend " + expected[provider] in logs, "Wrong active provider"
        else:
            assert "next backend CPU" in logs, "Auto fallback did not reach CPU"
        result = {"passed": True, "provider": provider, "alias": alias,
                  "cwd_has_spaces": True, "tests": tests, "clean_quit": True}
        (output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
        return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--package", type=Path, required=True)
    p.add_argument("--provider", choices=["cpu", "cuda", "tensorrt", "auto"], required=True)
    p.add_argument("--output", type=Path, default=Path("output/uci-release"))
    p.add_argument("--alias", action="store_true")
    a = p.parse_args()
    print(json.dumps(check(a.package.resolve(), a.provider, a.output.resolve(), a.alias), indent=2))
