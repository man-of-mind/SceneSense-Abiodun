// Record-only RFsim counters. One mmap per device/thread; no writer thread,
// payload logging, locks, or synchronous writes in the sample-processing loop.
#include <sys/mman.h>
#include <sys/syscall.h>
#include <fcntl.h>
#include <time.h>
#include <stdint.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace rd {
static constexpr unsigned N = 32, H = 10, P = 4;
struct Counter { uint64_t count, wall, cpu, max_wall, active_wall, hist[H]; };
struct Peer { int64_t fd; uint64_t local_port, remote_port, received_ts, required_ts;
  char model[64], local_ip[48], remote_ip[48]; };
struct Data { uint64_t seq, magic, pid, tid, ru, role, sample_rate, mono_origin, wall_origin,
  rx_ts, rx_calls, overflow; Peer peers[P]; Counter counters[N]; };
static uint64_t now(clockid_t id=CLOCK_MONOTONIC) {
  timespec t; if(clock_gettime(id,&t)) abort(); return uint64_t(t.tv_sec)*1000000000ULL+t.tv_nsec;
}
static void begin(Data* d) { __atomic_add_fetch(&d->seq,1,__ATOMIC_SEQ_CST); }
static void end(Data* d) { __atomic_add_fetch(&d->seq,1,__ATOMIC_SEQ_CST); }
struct Slot { const void* owner; Data* data; };
static thread_local Slot slots[4] = {};
static Data* data(rfsimulator_state_t* t) {
  for(auto& s:slots) if(s.owner==t) return s.data;
  const char* dir=getenv("M4_RFSIM_DIAGNOSTIC_DIR"); if(!dir) return nullptr;
  for(auto& s:slots) if(!s.owner) {
    char name[4096]; snprintf(name,sizeof(name),"%s/rfsim_%d_%ld_ru%d.bin",dir,getpid(),syscall(SYS_gettid),t->ru_id);
    int fd=open(name,O_RDWR|O_CREAT|O_EXCL|O_CLOEXEC,0644);
    if(fd<0 || ftruncate(fd,sizeof(Data))) { perror("RFsim diagnostic mmap create"); abort(); }
    auto* d=(Data*)mmap(nullptr,sizeof(Data),PROT_READ|PROT_WRITE,MAP_SHARED,fd,0);close(fd);
    if(d==MAP_FAILED) abort(); memset(d,0,sizeof(*d)); begin(d);
    d->magic=0x4d34524644494131ULL;d->pid=getpid();d->tid=syscall(SYS_gettid);
    d->ru=t->ru_id;d->role=t->role;d->sample_rate=t->sample_rate;
    d->mono_origin=now();d->wall_origin=now(CLOCK_REALTIME);end(d);
    s.owner=t;s.data=d;return d;
  }
  abort(); // Unknown device multiplicity must not silently alias counters.
}
static unsigned peer(Data* d,rfsimulator_state_t* t,buffer_t* b) {
  unsigned k=unsigned(b-t->buf); if(k>=P) { if(d){begin(d);d->overflow++;end(d);} return P; }
  if(!d) return k;
  begin(d);auto& p=d->peers[k];
  if(p.fd!=b->conn_sock || !p.local_port) {
    p.fd=b->conn_sock;
    if(b->channel_model && b->channel_model->model_name) snprintf(p.model,sizeof(p.model),"%s",b->channel_model->model_name);
    sockaddr_in a={};socklen_t n=sizeof(a);
    if(!getsockname(b->conn_sock,(sockaddr*)&a,&n)) { p.local_port=ntohs(a.sin_port);inet_ntop(AF_INET,&a.sin_addr,p.local_ip,sizeof(p.local_ip)); }
    n=sizeof(a);
    if(!getpeername(b->conn_sock,(sockaddr*)&a,&n)) { p.remote_port=ntohs(a.sin_port);inet_ntop(AF_INET,&a.sin_addr,p.remote_ip,sizeof(p.remote_ip)); }
  }
  p.received_ts=b->lastReceivedTS;end(d);return k;
}
// Counters are inclusive: READ contains WAIT, FLUSH contains EPOLL; PROCESS
// contains MERGE and CHANNEL. Never add nested or concurrent measurements.
struct Scope {
  Data* d; unsigned k;uint64_t w,c;
  Scope(Data* d_,unsigned k_):d(d_),k(k_),w(0),c(0) {
    if(!d||k>=N)return;w=now();c=now(CLOCK_THREAD_CPUTIME_ID);
    begin(d);d->counters[k].active_wall=w;end(d);
  }
  ~Scope() {
    if(!d||k>=N)return;uint64_t dc=now(CLOCK_THREAD_CPUTIME_ID)-c,dw=now()-w;
    const uint64_t limits[H-1]={100000,250000,500000,1000000,2000000,5000000,10000000,25000000,100000000};
    unsigned h=0;while(h<H-1&&dw>limits[h])++h;
    begin(d);auto& r=d->counters[k];r.count++;r.wall+=dw;r.cpu+=dc;
    if(dw>r.max_wall)r.max_wall=dw;r.hist[h]++;r.active_wall=0;end(d);
  }
};
static void progress(Data* d,rfsimulator_state_t* t) {
  if(!d)return;begin(d);d->rx_ts=t->nextRxTstamp;d->rx_calls++;end(d);
}
static unsigned waiting(Data* d,rfsimulator_state_t* t,buffer_t* b,int nsamps) {
  unsigned k=peer(d,t,b);if(d&&k<P){begin(d);d->peers[k].required_ts=t->nextRxTstamp+nsamps;end(d);}return k<P?8+k:N;
}
}
