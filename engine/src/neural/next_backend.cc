// New model backend. GPL-3.0-or-later, with the upstream Lc0 NVIDIA exception.
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#include "onnxruntime_cxx_api.h"
#include "neural/backend.h"
#include "neural/encoder.h"
#include "neural/register.h"
#include "neural/shared_params.h"
#include "neural/next_head_math.h"
#include "utils/commandline.h"

namespace lczero {
namespace {
constexpr size_t C=120, M=256, B=32;
struct Entry {
  std::array<float,64*C> x{};
  std::array<int64_t,M*4> moves{};
  std::array<bool,M> legal{};
  size_t count=0;
  int dump_id=-1;
  EvalResultPtr dst;
  std::string key;
};
struct Cached {float q=0,d=1,m=0;std::vector<float> p,aq,ad;};

Entry Encode(const EvalPosition& pos,EvalResultPtr dst,bool dump=true) {
  if(pos.pos.empty() || pos.legal_moves.size()>M)throw Exception("next: invalid evaluation shape");
  Entry e;e.count=pos.legal_moves.size();e.dst=dst;
  int transform=0;
  const auto planes=EncodePositionForNN(pblczero::NetworkFormat::INPUT_CLASSICAL_112_PLANE,
                                      pos.pos,8,FillEmptyHistory::NO,&transform);
  auto bit=[&](int col,int sq){return bool(planes[col].mask&(uint64_t(1)<<sq));};
  for(size_t c=0;c<104;c++){
    auto mask=planes[c].mask;e.key.append(reinterpret_cast<const char*>(&mask),sizeof(mask));
    for(size_t sq=0;sq<64;sq++)e.x[sq*C+c]=bit(c,sq)?1.f:0.f;
  }
  const auto& position=pos.pos.back();const auto& castle=position.GetBoard().castlings();
  const bool rights[4]={castle.we_can_000(),castle.we_can_00(),castle.they_can_000(),castle.they_can_00()};
  const int files[4]={castle.our_queenside_rook.idx,castle.our_kingside_rook.idx,
                      castle.their_queenside_rook.idx,castle.their_kingside_rook.idx};
  for(int c=0;c<4;c++){
    uint64_t mask=0;if(rights[c]){const int sq=(c<2?0:56)+files[c];e.x[sq*C+104+c]=1;mask=uint64_t(1)<<sq;}
    e.key.append(reinterpret_cast<const char*>(&mask),sizeof(mask));
  }
  const float stm=position.IsBlackToMove()?1.f:0.f,clock=position.GetRule50Ply()/100.f;
  e.key.append(reinterpret_cast<const char*>(&stm),sizeof(stm));e.key.append(reinterpret_cast<const char*>(&clock),sizeof(clock));
  for(int sq=0;sq<64;sq++){e.x[sq*C+108]=stm;e.x[sq*C+109]=clock;e.x[sq*C+119]=1;}
  for(int f=0;f<8;f++)if(bit(19,48+f)&&!bit(6,48+f)&&!bit(19,32+f)&&bit(6,32+f))e.x[(40+f)*C+110]=1;
  for(int h=0;h<8;h++){
    uint64_t occupied=0;for(int c=0;c<12;c++)occupied|=planes[h*13+c].mask;
    for(int sq=0;sq<64;sq++)e.x[sq*C+111+h]=occupied?1.f:0.f;
  }
  for(size_t j=0;j<e.count;j++){
    const auto move=pos.legal_moves[j];e.moves[4*j]=move.from().as_idx();e.moves[4*j+1]=move.to().as_idx();
    e.moves[4*j+2]=move.is_promotion()?move.promotion().idx+1:0;
    e.moves[4*j+3]=move.is_castling()?(move.to().file()>move.from().file()?1:2):0;e.legal[j]=true;
  }
  e.key.append(reinterpret_cast<const char*>(e.moves.data()),e.count*4*sizeof(int64_t));
  if(const char* directory=dump?std::getenv("NEXT_DUMP_INPUTS"):nullptr){
    static std::mutex mutex;static unsigned counter=0;std::lock_guard lock(mutex);
    e.dump_id=counter++;
    std::ofstream f(std::string(directory)+"/"+std::to_string(e.dump_id)+".bin",std::ios::binary);
    f.write(reinterpret_cast<const char*>(e.x.data()),sizeof(e.x));f.write(reinterpret_cast<const char*>(e.moves.data()),sizeof(e.moves));
    f.write(reinterpret_cast<const char*>(e.legal.data()),sizeof(e.legal));
  }
  return e;
}

void Trace(const Entry& e,const Cached& value){
  const char* directory=std::getenv("NEXT_DUMP_INPUTS");
  if(e.dump_id<0||!directory)return;
  std::ofstream f(std::string(directory)+"/"+std::to_string(e.dump_id)+".values",std::ios::binary);
  const float state[]={value.q,value.d,value.m};
  f.write(reinterpret_cast<const char*>(state),sizeof(state));
  for(const auto* values:{&value.p,&value.aq,&value.ad})
    f.write(reinterpret_cast<const char*>(values->data()),values->size()*sizeof(float));
}

void Copy(const Cached& c,const EvalResultPtr& dst){
  if(dst.q)*dst.q=c.q;
  if(dst.d)*dst.d=c.d;
  if(dst.m)*dst.m=c.m;
  if(!dst.p.empty())std::copy(c.p.begin(),c.p.end(),dst.p.begin());
  if(!dst.aq.empty())std::copy(c.aq.begin(),c.aq.end(),dst.aq.begin());
  if(!dst.ad.empty())std::copy(c.ad.begin(),c.ad.end(),dst.ad.begin());
}
std::array<float,3> Wdl(const float* p){
  float max=*std::max_element(p,p+3),sum=0;std::array<float,3> result;
  for(int j=0;j<3;j++){if(!std::isfinite(p[j]))throw Exception("next: nonfinite WDL");result[j]=std::exp(p[j]-max);sum+=result[j];}
  for(auto& x:result)x/=sum;
  return result;
}

void AppendTensorRT(Ort::SessionOptions& options,const std::string& path,bool capture) {
  OrtTensorRTProviderOptionsV2* raw=nullptr;
  Ort::ThrowOnError(Ort::GetApi().CreateTensorRTProviderOptions(&raw));
  std::unique_ptr<OrtTensorRTProviderOptionsV2,void(*)(OrtTensorRTProviderOptionsV2*)> guard(
      raw,[](auto* p){Ort::GetApi().ReleaseTensorRTProviderOptions(p);});
  const auto cache=path+".trt-cache";std::filesystem::create_directories(cache);
  const char* keys[]={"device_id","trt_fp16_enable","trt_max_workspace_size","trt_engine_cache_enable",
      "trt_engine_cache_path","trt_timing_cache_enable","trt_timing_cache_path","trt_profile_min_shapes",
      "trt_profile_opt_shapes","trt_profile_max_shapes","trt_cuda_graph_enable","trt_auxiliary_streams"};
  const char* values[]={"0","1","536870912","1",cache.c_str(),"1",cache.c_str(),
      "features:1x64x120,moves:1x256x4,legal:1x256",
      "features:8x64x120,moves:8x256x4,legal:8x256",
      "features:32x64x120,moves:32x256x4,legal:32x256",capture?"1":"0","0"};
  Ort::ThrowOnError(Ort::GetApi().UpdateTensorRTProviderOptions(raw,keys,values,12));
  options.AppendExecutionProvider_TensorRT_V2(*raw);
}

// A prebuilt engine wrapped as an EPContext model (model.trt.onnx) carries its
// own precision, optimization profile and tactics; only the device and graph
// replay are chosen here. Nothing is rebuilt at load time.
void AppendTensorRTContext(Ort::SessionOptions& options,bool capture,void* stream=nullptr) {
  OrtTensorRTProviderOptionsV2* raw=nullptr;
  Ort::ThrowOnError(Ort::GetApi().CreateTensorRTProviderOptions(&raw));
  std::unique_ptr<OrtTensorRTProviderOptionsV2,void(*)(OrtTensorRTProviderOptionsV2*)> guard(
      raw,[](auto* p){Ort::GetApi().ReleaseTensorRTProviderOptions(p);});
  const char* keys[]={"device_id","trt_cuda_graph_enable","trt_auxiliary_streams"};
  const char* values[]={"0",capture?"1":"0","0"};
  Ort::ThrowOnError(Ort::GetApi().UpdateTensorRTProviderOptions(raw,keys,values,3));
  if(stream) Ort::ThrowOnError(Ort::GetApi().UpdateTensorRTProviderOptionsWithValue(raw,"user_compute_stream",stream));
  options.AppendExecutionProvider_TensorRT_V2(*raw);
}

// Zeroed page-locked host memory, so queued uploads need no staging copy.
std::shared_ptr<void> PinnedRows(size_t bytes) {
  void* runtime=dlopen("libcudart.so.13",RTLD_NOW|RTLD_LOCAL);
  if(!runtime) throw Exception("next: CUDA runtime unavailable for pinned rows");
  using AllocHostFn=int(*)(void**,size_t,unsigned);using FreeHostFn=int(*)(void*);
  auto allocate=reinterpret_cast<AllocHostFn>(dlsym(runtime,"cudaHostAlloc"));
  auto release=reinterpret_cast<FreeHostFn>(dlsym(runtime,"cudaFreeHost"));
  void* p=nullptr;
  if(!allocate||!release||allocate(&p,bytes,0)){dlclose(runtime);throw Exception("next: pinned host allocation failed");}
  std::memset(p,0,bytes);
  return std::shared_ptr<void>(p,[release,runtime](void* q){release(q);dlclose(runtime);});
}

// One session owns all captured batch sizes. Each graph ID has stable input
// and output buffers. The enclosing run_mutex serializes capture and replay.
// TensorRT EP supports only one capture: its session always executes eight
// independent rows. Requests above eight use the separate non-captured session.
class CudaGraphs {
 public:
  CudaGraphs(Ort::Env& env,const std::string& path,float* x,int64_t* moves,bool* legal,bool tensorrt,bool context,
             bool async,std::array<bool,5> output_mask,bool allow_full=true)
      : runtime_(dlopen("libcudart.so.13",RTLD_NOW|RTLD_LOCAL),dlclose),tensorrt_(tensorrt),output_mask_(output_mask) {
    if(!runtime_) throw Exception("next: CUDA runtime unavailable for graph copies");
    copy_=reinterpret_cast<CopyFn>(dlsym(runtime_.get(),"cudaMemcpy"));
    if(!copy_) throw Exception("next: cudaMemcpy unavailable");
    // A pipelined TensorRT lane owns a non-blocking stream: its copies and
    // replay never wait on another lane or on the legacy default stream.
    async_=async&&tensorrt&&context;
    const char* full_io=std::getenv("NEXT_TRT_FULL_GRAPH");
    const bool capture_io=async_&&allow_full&&(!full_io||std::string(full_io)!="0");
    if(async_) {
      copy_async_=Load<CopyAsyncFn>("cudaMemcpyAsync");
      sync_=Load<UnaryFn>("cudaStreamSynchronize");
      void* stream=nullptr;
      Check(Load<CreateStreamFn>("cudaStreamCreateWithFlags")(&stream,1)); // cudaStreamNonBlocking
      stream_=Resource(stream,Load<UnaryFn>("cudaStreamDestroy"));
      allocate_host_=Load<AllocHostFn>("cudaHostAlloc");
      free_host_=Load<UnaryFn>("cudaFreeHost");
    }
    Ort::SessionOptions options;
    options.SetIntraOpNumThreads(1);options.SetInterOpNumThreads(1);
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    if(tensorrt) context?AppendTensorRTContext(options,!capture_io,stream_.get()):AppendTensorRT(options,path,true);
    OrtCUDAProviderOptionsV2* raw=nullptr;
    Ort::ThrowOnError(Ort::GetApi().CreateCUDAProviderOptions(&raw));
    std::unique_ptr<OrtCUDAProviderOptionsV2,void(*)(OrtCUDAProviderOptionsV2*)> cuda(
        raw,[](auto* p){Ort::GetApi().ReleaseCUDAProviderOptions(p);});
    const char* keys[]={"device_id","gpu_mem_limit","arena_extend_strategy","enable_cuda_graph"};
    const char* values[]={"0","1073741824","kSameAsRequested",tensorrt?"0":"1"};
    Ort::ThrowOnError(Ort::GetApi().UpdateCUDAProviderOptions(raw,keys,values,4));
    if(async_) Ort::ThrowOnError(Ort::GetApi().UpdateCUDAProviderOptionsWithValue(raw,"user_compute_stream",stream_.get()));
    options.AppendExecutionProvider_CUDA_V2(*raw);
    session_=std::make_unique<Ort::Session>(env,path.c_str(),options);
    Ort::MemoryInfo device("Cuda",OrtArenaAllocator,0,OrtMemTypeDefault);
    allocator_=std::make_unique<Ort::Allocator>(*session_,device);
    auto host=Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,OrtMemTypeDefault);
    Ort::AllocatorWithDefaultOptions cpu;
    for(size_t n=tensorrt?8:1;n<=(tensorrt?8:B);n++) {
      auto& slot=slots_[n];slot.binding=std::make_unique<Ort::IoBinding>(*session_);
      slot.options.AddConfigEntry("gpu_graph_id",std::to_string(n).c_str());
      // Run() then only queues the replay; the lane's stream wait completes it.
      if(async_) slot.options.AddConfigEntry("disable_synchronize_execution_providers","1");
      const int64_t xs[]={int64_t(n),64,C},ms[]={int64_t(n),M,4},ls[]={int64_t(n),M};
      slot.host_inputs.push_back(Ort::Value::CreateTensor<float>(host,x,n*64*C,xs,3));
      slot.host_inputs.push_back(Ort::Value::CreateTensor<int64_t>(host,moves,n*M*4,ms,3));
      slot.host_inputs.push_back(Ort::Value::CreateTensor<bool>(host,legal,n*M,ls,2));
      slot.device_inputs.push_back(Ort::Value::CreateTensor<float>(*allocator_,xs,3));
      slot.device_inputs.push_back(Ort::Value::CreateTensor<int64_t>(*allocator_,ms,3));
      slot.device_inputs.push_back(Ort::Value::CreateTensor<bool>(*allocator_,ls,2));
      const char* ins[]={"features","moves","legal"};
      for(size_t i=0;i<3;i++) slot.binding->BindInput(ins[i],slot.device_inputs[i]);
      const char* outs[]={"policy_logits","wdl_logits","action_wdl_logits","duration_params","outcome_logits"};
      const std::array<std::vector<int64_t>,5> shapes={{{int64_t(n),M},{int64_t(n),3},
          {int64_t(n),M,3},{int64_t(n),2},{int64_t(n),3}}};
      for(size_t i=0;i<5;i++) {
        slot.device_outputs.push_back(Ort::Value::CreateTensor<float>(*allocator_,shapes[i].data(),shapes[i].size()));
        if(async_) {
          size_t elements=1;for(auto size:shapes[i])elements*=size;
          slot.pinned_outputs.push_back(Pinned(elements*sizeof(float)));
          slot.host_outputs.push_back(Ort::Value::CreateTensor<float>(host,
              static_cast<float*>(slot.pinned_outputs.back().get()),elements,shapes[i].data(),shapes[i].size()));
        } else slot.host_outputs.push_back(Ort::Value::CreateTensor<float>(cpu,shapes[i].data(),shapes[i].size()));
        slot.binding->BindOutput(outs[i],slot.device_outputs.back());
      }
      Run(n); // Capture every shape before readyok, outside the move clock.
      if(tensorrt) for(int warm=0;warm<3;warm++) Run(n);
    }
    if(capture_io) {
      try {CaptureIo();}
      catch(const std::exception& error) {
        throw Exception(std::string("next: full-transfer capture failed: ")+error.what());
      }
    }
  }
  ~CudaGraphs() {
    // Drain queued copies and replays, including after an exception, before
    // any buffer or the session they reference is released.
    if(async_) sync_(stream_.get());
    io_exec_.reset(); // Captured pointers must not outlive the tensors/session.
  }
  const std::vector<Ort::Value>& Run(size_t n) {
    if(n<1||n>B||(tensorrt_&&n!=8)) throw Exception("next: invalid captured batch size");
    auto& slot=slots_.at(n);
    const std::array<size_t,3> input_bytes={n*64*C*sizeof(float),n*M*4*sizeof(int64_t),n*M*sizeof(bool)};
    const std::array<size_t,5> output_elements={n*M,n*3,n*M*3,n*2,n*3};
    if(async_) {
      try {
        if(io_exec_) Check(graph_launch_(io_exec_.get(),stream_.get()));
        else EnqueueIo(n);
        Check(sync_(stream_.get()));
      } catch(...) {
        sync_(stream_.get());
        throw;
      }
      return slot.host_outputs;
    }
    for(size_t i=0;i<3;i++) Check(copy_(slot.device_inputs[i].GetTensorMutableRawData(),
        slot.host_inputs[i].GetTensorRawData(),input_bytes[i],1)); // host -> device
    session_->Run(slot.options,*slot.binding);
    // Copy exactly the head outputs required by the active configuration.
    for(size_t i=0;i<5;i++) if(output_mask_[i]) Check(copy_(slot.host_outputs[i].GetTensorMutableRawData(),
        slot.device_outputs[i].GetTensorRawData(),output_elements[i]*sizeof(float),2)); // device -> host
    return slot.host_outputs;
  }
 private:
  using CopyFn=int(*)(void*,const void*,size_t,int);
  using CopyAsyncFn=int(*)(void*,const void*,size_t,int,void*);
  using UnaryFn=int(*)(void*);
  using CreateStreamFn=int(*)(void**,unsigned);
  using AllocHostFn=int(*)(void**,size_t,unsigned);
  using BeginCaptureFn=int(*)(void*,int);
  using EndCaptureFn=int(*)(void*,void**);
  using InstantiateFn=int(*)(void**,void*,unsigned long long);
  using LaunchFn=int(*)(void*,void*);
  using Resource=std::unique_ptr<void,UnaryFn>;
  static void Check(int code){if(code)throw Exception("next: CUDA tensor copy failed: "+std::to_string(code));}
  template<class T> T Load(const char* name) {
    auto fn=reinterpret_cast<T>(dlsym(runtime_.get(),name));
    if(!fn) throw Exception(std::string("next: missing CUDA function ")+name);
    return fn;
  }
  Resource Pinned(size_t bytes) {void* p=nullptr;Check(allocate_host_(&p,bytes,0));return Resource(p,free_host_);}
  void EnqueueIo(size_t n) {
    auto& slot=slots_.at(n);
    const std::array<size_t,3> bytes={n*64*C*sizeof(float),n*M*4*sizeof(int64_t),n*M*sizeof(bool)};
    const std::array<size_t,5> elements={n*M,n*3,n*M*3,n*2,n*3};
    for(size_t i=0;i<3;i++) Check(copy_async_(slot.device_inputs[i].GetTensorMutableRawData(),
        slot.host_inputs[i].GetTensorRawData(),bytes[i],1,stream_.get()));
    session_->Run(slot.options,*slot.binding);
    for(size_t i=0;i<5;i++) if(output_mask_[i]) Check(copy_async_(slot.host_outputs[i].GetTensorMutableRawData(),
        slot.device_outputs[i].GetTensorRawData(),elements[i]*sizeof(float),2,stream_.get()));
  }
  void CaptureIo() {
    const auto begin=Load<BeginCaptureFn>("cudaStreamBeginCapture");
    const auto end=Load<EndCaptureFn>("cudaStreamEndCapture");
    const auto destroy=Load<UnaryFn>("cudaGraphDestroy");
    graph_launch_=Load<LaunchFn>("cudaGraphLaunch");
    void* raw=nullptr;
    Check(begin(stream_.get(),1)); // thread-local capture; other lanes are independent
    try {EnqueueIo(8);} catch(...) {
      end(stream_.get(),&raw);if(raw)destroy(raw);throw;
    }
    const int status=end(stream_.get(),&raw);
    Resource graph(raw,destroy);Check(status);
    void* executable=nullptr;
    Check(Load<InstantiateFn>("cudaGraphInstantiateWithFlags")(&executable,graph.get(),0));
    io_exec_=Resource(executable,Load<UnaryFn>("cudaGraphExecDestroy"));
    Run(8); // Check replay before readiness and before the GUI clock starts.
    std::cout<<"info string next TensorRT full-transfer graph ready"<<std::endl;
  }
  struct Slot {
    Ort::RunOptions options;
    std::vector<Ort::Value> host_inputs,device_inputs,host_outputs,device_outputs;
    std::vector<Resource> pinned_outputs;
    std::unique_ptr<Ort::IoBinding> binding;
  };
  std::unique_ptr<void,int(*)(void*)> runtime_;
  CopyFn copy_=nullptr;
  CopyAsyncFn copy_async_=nullptr;
  UnaryFn sync_=nullptr,free_host_=nullptr;
  AllocHostFn allocate_host_=nullptr;
  bool async_=false;
  Resource stream_{nullptr,nullptr};
  bool tensorrt_=false;
  std::array<bool,5> output_mask_;
  std::unique_ptr<Ort::Session> session_;
  std::unique_ptr<Ort::Allocator> allocator_;
  std::array<Slot,B+1> slots_;
  LaunchFn graph_launch_=nullptr;
  Resource io_exec_{nullptr,nullptr};
};

// ORT stores graph captures in its CUDA per-thread context. Search workers are
// recreated every move; capture, replay and destruction must stay on one worker
// to keep readiness warm-up effective and avoid recapturing on the move clock.
class GraphWorker {
 public:
  GraphWorker(Ort::Env& env,const std::string& path,float* x,int64_t* moves,bool* legal,bool tensorrt,bool context,
              bool async,std::array<bool,5> output_mask) {
    worker_=std::thread([this,&env,path,x,moves,legal,tensorrt,context,async,output_mask] {
      std::unique_ptr<CudaGraphs> graph;
      try {
        try {graph=std::make_unique<CudaGraphs>(env,path,x,moves,legal,tensorrt,context,async,output_mask);}
        catch(const std::exception& error) {
          if(std::string(error.what()).find("next: full-transfer capture failed:")!=0)throw;
          std::cout<<"info string next full-transfer capture unavailable; retaining inference-only graphs"<<std::endl;
          graph=std::make_unique<CudaGraphs>(env,path,x,moves,legal,tensorrt,context,async,output_mask,false);
        }
      }
      catch(...) {error_=std::current_exception();}
      std::unique_lock lock(mutex_);ready_=true;completed_.notify_one();
      if(error_) return;
      for(;;) {
        requested_.wait(lock,[this]{return stop_||pending_;});
        if(stop_) break;
        const size_t n=pending_;pending_=0;lock.unlock();
        const std::vector<Ort::Value>* output=nullptr;std::exception_ptr error;
        try {output=&graph->Run(n);} catch(...) {error=std::current_exception();}
        lock.lock();output_=output;error_=error;done_=true;completed_.notify_one();
      }
    });
    std::unique_lock lock(mutex_);completed_.wait(lock,[this]{return ready_;});
    if(error_) {lock.unlock();worker_.join();std::rethrow_exception(error_);}
  }
  ~GraphWorker() {
    {std::lock_guard lock(mutex_);stop_=true;requested_.notify_one();}
    worker_.join();
  }
  const std::vector<Ort::Value>& Run(size_t n) {
    std::unique_lock lock(mutex_);pending_=n;done_=false;requested_.notify_one();
    completed_.wait(lock,[this]{return done_;});
    if(error_) std::rethrow_exception(error_);
    return *output_;
  }
 private:
  std::mutex mutex_;
  std::condition_variable requested_,completed_;
  bool ready_=false,done_=false,stop_=false;
  size_t pending_=0;
  std::exception_ptr error_;
  const std::vector<Ort::Value>* output_=nullptr;
  std::thread worker_;
};

class NextBackend;
class Computation:public BackendComputation {
 public:
  explicit Computation(NextBackend* b):backend(b){entries.reserve(B);}
  size_t UsedBatchSize() const override {std::lock_guard lock(mutex);return entries.size();}
  AddInputResult AddInput(const EvalPosition&,EvalResultPtr) override;
  void ComputeBlocking() override;
 private:
  NextBackend* backend;mutable std::mutex mutex;std::vector<Entry> entries;
};

class NextBackend:public Backend {
 public:
  explicit NextBackend(const OptionsDict& options):env(ORT_LOGGING_LEVEL_WARNING,"next"){
    env.DisableTelemetryEvents();
    auto path=options.Get<std::string>(SharedBackendParams::kWeightsId);
    if(path.empty()||path[0]=='<')path=CommandLine::BinaryDirectory()+"/model.onnx";
    // A prebuilt TensorRT engine for exactly this model sits beside it as
    // <model>.trt.onnx. It is tried first; CUDA and CPU keep the ordinary model.
    const auto trt_model=std::filesystem::path(path).replace_extension(".trt.onnx").string();
    trt_context=std::filesystem::exists(trt_model);
    auto session_model=path;
    const char* requested=std::getenv("NEXT_PROVIDER");std::string provider=requested?requested:"auto";
    if(provider!="auto"&&provider!="cpu"&&provider!="cuda"&&provider!="tensorrt")throw Exception("next: unknown provider");
    enable_action=options.Get<bool>(SharedBackendParams::kNextActionValues);
    enable_duration=options.Get<bool>(SharedBackendParams::kNextMovesLeft);
    enable_outcome=options.Get<bool>(SharedBackendParams::kNextOutcomeBlend);
    next_heads::moves_left_activated.store(false,std::memory_order_relaxed);
    action_outputs=enable_action||std::getenv("NEXT_DUMP_INPUTS")!=nullptr;
    policy_temperature=options.Get<float>(SharedBackendParams::kPolicySoftmaxTemp);
    if(!std::isfinite(policy_temperature)||policy_temperature<=0)
      throw Exception("next: policy temperature must be finite and positive");
    profile=std::getenv("NEXT_PROFILE")!=nullptr;
    auto make_options=[](){Ort::SessionOptions o;o.SetIntraOpNumThreads(2);o.SetInterOpNumThreads(1);o.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);return o;};
    auto opts=make_options();bool gpu=false;
    if(provider!="cpu"){
      std::vector<std::string> attempts;
      if(provider=="auto") {
        if(trt_context||std::filesystem::exists(CommandLine::BinaryDirectory()+"/tensorrt.default")) attempts.push_back("tensorrt");
        attempts.push_back("cuda");
      } else attempts.push_back(provider);
      for(const auto& attempt:attempts) {
      try {
        // Optional adjacent libraries make a GUI launch independent of the
        // shell's LD_LIBRARY_PATH and of importing Python/PyTorch first.
        dlopen((CommandLine::BinaryDirectory()+"/lib/libcudnn.so.9").c_str(),RTLD_NOW|RTLD_GLOBAL);
        if(attempt=="tensorrt") {
          const auto directory=CommandLine::BinaryDirectory()+"/lib/";
          for(const char* name:{"libnvinfer.so.10","libnvinfer_plugin.so.10","libnvonnxparser.so.10"})
            dlopen((directory+name).c_str(),RTLD_NOW|RTLD_GLOBAL);
          if(trt_context) AppendTensorRTContext(opts,false); else AppendTensorRT(opts,path,false);
        }
        OrtCUDAProviderOptionsV2* cuda=nullptr;Ort::ThrowOnError(Ort::GetApi().CreateCUDAProviderOptions(&cuda));
        std::unique_ptr<OrtCUDAProviderOptionsV2,void(*)(OrtCUDAProviderOptionsV2*)> guard(cuda,[](auto* p){Ort::GetApi().ReleaseCUDAProviderOptions(p);});
        const char* keys[]={"device_id","gpu_mem_limit","arena_extend_strategy"};const char* values[]={"0","2147483648","kSameAsRequested"};
        Ort::ThrowOnError(Ort::GetApi().UpdateCUDAProviderOptions(cuda,keys,values,3));opts.AppendExecutionProvider_CUDA_V2(*cuda);
        const auto& load=attempt=="tensorrt"&&trt_context?trt_model:path;
        session=std::make_unique<Ort::Session>(env,load.c_str(),opts);session_model=load;gpu=true;trt_graph=attempt=="tensorrt";break;
      } catch(const std::exception& e){if(provider!="auto")throw;std::cerr<<"next "<<attempt<<" unavailable: "<<e.what()<<'\n';opts=make_options();}
      }
    }
    if(!session)session=std::make_unique<Ort::Session>(env,path.c_str(),opts);
    if(session->GetInputCount()!=3||session->GetOutputCount()!=5)throw Exception("next: incompatible model I/O schema");
    Ort::AllocatorWithDefaultOptions allocator;
    auto schema=session->GetModelMetadata().LookupCustomMetadataMapAllocated("schema",allocator);
    if(!schema||std::string(schema.get())!="\"nextchess-v1\"")throw Exception("next: unsupported model contract");
    auto config=session->GetModelMetadata().LookupCustomMetadataMapAllocated("config",allocator);
    if(enable_action&&(!config||std::string(config.get()).find("\"action_values\": true")==std::string::npos))
      throw Exception("next: action-value search requested for a model without trained action values");
    const std::array<std::vector<int64_t>,3> input_shapes={{{-1,64,C},{-1,M,4},{-1,M}}};
    const std::array<ONNXTensorElementDataType,3> input_types={ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL};
    for(size_t i=0;i<3;i++){
      auto type=session->GetInputTypeInfo(i);auto tensor=type.GetTensorTypeAndShapeInfo();
      if(tensor.GetShape()!=input_shapes[i]||tensor.GetElementType()!=input_types[i])throw Exception("next: invalid input tensor contract");
    }
    attrs={.has_mlh=enable_duration,.has_wdl=true,.runs_on_cpu=!gpu,.suggested_num_search_threads=2,
           .recommended_batch_size=8,.maximum_batch_size=1024,.has_action_values=enable_action,.has_own_input_cache=true};
    Backend::UpdateConfiguration(options);
    std::cout<<"info string next backend "<<(gpu?(trt_graph?(trt_context?"TensorRT engine":"TensorRT FP16"):"CUDA"):"CPU")<<"; action values "<<(enable_action?"experimental":"off")<<std::endl;
    std::cout<<"info string next heads action_values="<<enable_action
             <<" moves_left="<<enable_duration<<" outcome_blend="<<enable_outcome<<std::endl;
    // Warm up before the GUI receives readyok, outside its move clock.
    Entry dummy;dummy.count=1;dummy.legal[0]=true;Run(std::span<const Entry>(&dummy,1));
    const char* graph_option=std::getenv("NEXT_CUDA_GRAPHS");
    if(gpu && (!graph_option||std::string(graph_option)!="0")) {
      const std::array<bool,5> mask{true,true,action_outputs,enable_duration,enable_outcome};
      try {
        // All rows are independent. Warm-up rows have one legal move so that
        // every output is finite, including rows outside the first batch.
        for(size_t i=0;i<B;i++) legal_buffer[i*M]=true;
        if(trt_graph&&trt_context) {
          const char* option=std::getenv("NEXT_TRT_LANES");
          const int count=std::clamp(option?std::atoi(option):2,1,4);
          const size_t xb=B*64*C*sizeof(float),mb=B*M*4*sizeof(int64_t),lb=B*M*sizeof(bool);
          for(int i=0;i<count;i++) {
            auto lane=std::make_unique<Lane>();
            lane->pinned=PinnedRows(xb+mb+lb);
            auto* bytes=static_cast<char*>(lane->pinned.get());
            lane->x=reinterpret_cast<float*>(bytes);lane->moves=reinterpret_cast<int64_t*>(bytes+xb);
            lane->legal=reinterpret_cast<bool*>(bytes+xb+mb);
            for(size_t r=0;r<B;r++) lane->legal[r*M]=true;
            lane->worker=std::make_unique<GraphWorker>(env,session_model,lane->x,lane->moves,lane->legal,true,true,true,mask);
            lanes.push_back(std::move(lane));
          }
          std::cout<<"info string next TensorRT graph ready: fixed batch 8; requests 1-8 padded; larger batches uncaptured; "
                   <<count<<" pipelined lane"<<(count>1?"s":"")<<", pinned asynchronous transfers"<<std::endl;
        } else {
          graphs=std::make_unique<GraphWorker>(env,session_model,x_buffer.data(),move_buffer.data(),legal_buffer.data(),trt_graph,
              trt_graph&&trt_context,false,mask);
          std::cout<<(trt_graph?"info string next TensorRT graph ready: fixed batch 8; requests 1-8 padded; larger batches uncaptured":
                                "info string next CUDA graphs ready: all batches 1-32; stable device buffers")<<std::endl;
        }
      } catch(const std::exception& error) {
        lanes.clear();graphs.reset();
        std::cout<<"info string next graph unavailable; ordinary inference retained: "<<error.what()<<std::endl;
      }
    }
  }
  ~NextBackend() override {
    std::cerr<<"next heads summary action_values="<<enable_action<<" moves_left="<<enable_duration
             <<" outcome_blend="<<enable_outcome<<" moves_left_activated="
             <<next_heads::moves_left_activated.load(std::memory_order_relaxed)<<'\n';
    if(profile) for(size_t n=1;n<=B;n++) if(batch_calls[n])
      std::cerr<<"next profile batch "<<n<<" calls "<<batch_calls[n]
               <<" inference_us "<<batch_us[n]<<" graph "<<(graphs&&(!trt_graph||n<=8))<<'\n';
  }
  BackendAttributes GetAttributes() const override{return attrs;}
  std::unique_ptr<BackendComputation> CreateComputation() override{return std::make_unique<Computation>(this);}
  std::optional<EvalResult> GetCachedEvaluation(const EvalPosition& pos) override{
    if(pos.pos.empty())return std::nullopt;
    MoveList generated;
    EvalPosition request=pos;
    if(request.legal_moves.empty()){
      generated=pos.pos.back().GetBoard().GenerateLegalMoves();
      if(generated.empty())return std::nullopt;
      request.legal_moves=generated;
    }
    auto e=Encode(request,{},false);
    std::lock_guard lock(cache_mutex);auto it=cache.find(e.key);
    if(it==cache.end())return std::nullopt;
    const auto& c=it->second;EvalResult result;result.q=c.q;result.d=c.d;result.m=c.m;
    if(!pos.legal_moves.empty()){result.p=c.p;result.aq=c.aq;result.ad=c.ad;}
    return result;
  }
  UpdateConfigurationResult UpdateConfiguration(const OptionsDict& options)override{
    return IsSameConfiguration(options)?UPDATE_OK:NEED_RESTART;
  }
  std::vector<Cached> Run(std::span<const Entry> rows){
    const size_t n=rows.size();
    if(n<1||n>B)throw Exception("next: inference batch exceeds preallocated buffers");
    if(!lanes.empty()&&n<=8) return RunLane(rows);
    std::lock_guard lock(run_mutex);
    auto& x=x_buffer;auto& moves=move_buffer;auto& legal=legal_buffer;
    for(size_t i=0;i<n;i++){
      std::copy(rows[i].x.begin(),rows[i].x.end(),x.begin()+i*64*C);
      std::copy(rows[i].moves.begin(),rows[i].moves.end(),moves.begin()+i*M*4);
      std::copy(rows[i].legal.begin(),rows[i].legal.end(),legal.data()+i*M);
    }
    const int64_t xs[]={int64_t(n),64,C},ms[]={int64_t(n),M,4},ls[]={int64_t(n),M};
    auto memory=Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,OrtMemTypeDefault);
    std::vector<Ort::Value> inputs;
    inputs.reserve(3);
    inputs.push_back(Ort::Value::CreateTensor<float>(memory,x.data(),n*64*C,xs,3));
    inputs.push_back(Ort::Value::CreateTensor<int64_t>(memory,moves.data(),n*M*4,ms,3));
    inputs.push_back(Ort::Value::CreateTensor<bool>(memory,legal.data(),n*M,ls,2));
    const char* in_names[]={"features","moves","legal"};const char* out_names[]={"policy_logits","wdl_logits","action_wdl_logits","duration_params","outcome_logits"};
    const auto started=std::chrono::steady_clock::now();
    std::vector<Ort::Value> dynamic_outputs;
    const bool use_graph=graphs&&(!trt_graph||n<=8);
    const size_t output_rows=use_graph&&trt_graph?8:n;
    if(!use_graph) dynamic_outputs=session->Run(Ort::RunOptions{nullptr},in_names,inputs.data(),3,out_names,5);
    const auto& outputs=use_graph?graphs->Run(output_rows):dynamic_outputs;
    Profile(n,started);
    return Decode(outputs,rows,output_rows);
  }
  // A pipelined lane: take an idle one if any, otherwise wait for the next in
  // turn. Rows are staged in the lane's pinned memory and decoded before the
  // lane is released, because its host outputs are reused by the next batch.
  std::vector<Cached> RunLane(std::span<const Entry> rows){
    Lane* lane=nullptr;std::unique_lock<std::mutex> hold;
    const unsigned first=next_lane.fetch_add(1,std::memory_order_relaxed);
    for(size_t k=0;k<lanes.size()&&!lane;k++){
      Lane* candidate=lanes[(first+k)%lanes.size()].get();
      std::unique_lock<std::mutex> attempt(candidate->mutex,std::try_to_lock);
      if(attempt.owns_lock()){lane=candidate;hold=std::move(attempt);}
    }
    if(!lane){lane=lanes[first%lanes.size()].get();hold=std::unique_lock<std::mutex>(lane->mutex);}
    const size_t n=rows.size();
    for(size_t i=0;i<n;i++){
      std::copy(rows[i].x.begin(),rows[i].x.end(),lane->x+i*64*C);
      std::copy(rows[i].moves.begin(),rows[i].moves.end(),lane->moves+i*M*4);
      std::copy(rows[i].legal.begin(),rows[i].legal.end(),lane->legal+i*M);
    }
    const auto started=std::chrono::steady_clock::now();
    const auto& outputs=lane->worker->Run(8);
    Profile(n,started);
    return Decode(outputs,rows,8);
  }
  void Profile(size_t n,std::chrono::steady_clock::time_point started){
    if(!profile)return;
    std::lock_guard lock(profile_mutex);
    batch_calls[n]++;
    batch_us[n]+=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-started).count();
  }
  std::vector<Cached> Decode(const std::vector<Ort::Value>& outputs,std::span<const Entry> rows,size_t output_rows) const{
    const size_t n=rows.size();
    const std::array<size_t,5> sizes={output_rows*M,output_rows*3,output_rows*M*3,output_rows*2,output_rows*3};
    for(size_t i=0;i<5;i++){
      auto info=outputs[i].GetTensorTypeAndShapeInfo();
      if(info.GetElementType()!=ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT||info.GetElementCount()!=sizes[i])throw Exception("next: invalid output tensor contract");
    }
    const float* policy=outputs[0].GetTensorData<float>();const float* state=outputs[1].GetTensorData<float>();
    const float* actions=outputs[2].GetTensorData<float>();std::vector<Cached> results(n);
    for(size_t i=0;i<n;i++){
      auto& r=results[i];auto wdl=Wdl(state+3*i);
      if(enable_outcome){auto outcome=Wdl(outputs[4].GetTensorData<float>()+3*i);
        for(size_t k=0;k<3;k++)wdl[k]=.75f*wdl[k]+.25f*outcome[k];}
      r.q=wdl[0]-wdl[2];r.d=wdl[1];
      if(enable_duration){const float* duration=outputs[3].GetTensorData<float>()+2*i;
        if(!std::isfinite(duration[0])||!std::isfinite(duration[1]))throw Exception("next: nonfinite duration");
        r.m=next_heads::MedianPlies(duration[0]);}
      r.p.resize(rows[i].count);
      if(action_outputs){r.aq.resize(rows[i].count);r.ad.resize(rows[i].count);}
      float max=-1e30f,sum=0;for(size_t j=0;j<r.p.size();j++)max=std::max(max,policy[i*M+j]);
      for(size_t j=0;j<r.p.size();j++){
        if(!std::isfinite(policy[i*M+j]))throw Exception("next: nonfinite policy");
        r.p[j]=std::exp((policy[i*M+j]-max)/policy_temperature);sum+=r.p[j];
        if(action_outputs){auto a=Wdl(actions+(i*M+j)*3);r.aq[j]=a[0]-a[2];r.ad[j]=a[1];}
      }
      // Lc0's value-only prefetch deliberately supplies no legal moves.
      // Its value is valid; there is no policy distribution to normalize.
      if(!r.p.empty()&&sum<=0)throw Exception("next: empty policy mass");
      for(auto& p:r.p)p/=sum;
    }
    return results;
  }
  std::mutex cache_mutex;std::unordered_map<std::string,Cached> cache;
 private:
  Ort::Env env;std::unique_ptr<Ort::Session> session;BackendAttributes attrs;
  bool enable_action=false,action_outputs=false,enable_duration=false,enable_outcome=false;
  float policy_temperature=1.f;
  std::mutex run_mutex;
  std::array<float,B*64*C> x_buffer{};
  std::array<int64_t,B*M*4> move_buffer{};
  std::array<bool,B*M> legal_buffer{};
  std::unique_ptr<GraphWorker> graphs;
  // Independent TensorRT inference lanes: each has its own session, captured
  // graph, non-blocking stream, worker thread and pinned rows, so one lane's
  // batch is staged and queued while another's runs and the GPU need not wait
  // between batches. NEXT_TRT_LANES (1-4, default 2) sets the count.
  struct Lane {
    std::mutex mutex;
    std::shared_ptr<void> pinned;  // declared before worker: outlives it
    float* x=nullptr;int64_t* moves=nullptr;bool* legal=nullptr;
    std::unique_ptr<GraphWorker> worker;
  };
  std::vector<std::unique_ptr<Lane>> lanes;
  std::atomic<unsigned> next_lane{0};
  bool trt_graph=false,trt_context=false;
  bool profile=false;
  std::mutex profile_mutex;
  std::array<uint64_t,B+1> batch_calls{};
  std::array<double,B+1> batch_us{};
};

BackendComputation::AddInputResult Computation::AddInput(const EvalPosition& pos,EvalResultPtr result){
  auto entry=Encode(pos,result);
  {std::lock_guard lock(backend->cache_mutex);auto it=backend->cache.find(entry.key);
   if(it!=backend->cache.end()){Copy(it->second,result);Trace(entry,it->second);return FETCHED_IMMEDIATELY;}}
  std::lock_guard lock(mutex);entries.push_back(std::move(entry));return ENQUEUED_FOR_EVAL;
}
void Computation::ComputeBlocking(){
  for(size_t start=0;start<entries.size();start+=B){
    auto span=std::span<const Entry>(entries.data()+start,std::min(B,entries.size()-start));auto result=backend->Run(span);
    for(size_t j=0;j<result.size();j++){
      Copy(result[j],span[j].dst);Trace(span[j],result[j]);std::lock_guard lock(backend->cache_mutex);
      if(backend->cache.size()>=4096)backend->cache.clear();
      backend->cache.emplace(span[j].key,std::move(result[j]));
    }
  }
}
class NextFactory:public BackendFactory{
 public:int GetPriority()const override{return 1000;}std::string_view GetName()const override{return "next-ort";}
  std::unique_ptr<Backend> Create(const OptionsDict& opts)override{return std::make_unique<NextBackend>(opts);}
};
static BackendManager::Register next_registration(std::make_unique<NextFactory>());
} // namespace
} // namespace lczero
