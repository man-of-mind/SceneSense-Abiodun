#include <sys/socket.h>
#include <sys/types.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <cassert>
#include <cstring>
struct channel {const char* model_name;};
struct buffer_t {int conn_sock;long lastReceivedTS;channel* channel_model;};
struct rfsimulator_state_t {int ru_id,role;double sample_rate;long nextRxTstamp;buffer_t buf[250];};
#include "diagnostic.h"
int main(int argc,char**argv){
 assert(argc==2);setenv("M4_RFSIM_DIAGNOSTIC_DIR",argv[1],1);
 int l=socket(AF_INET,SOCK_STREAM,0),c=socket(AF_INET,SOCK_STREAM,0);assert(l>=0&&c>=0);
 sockaddr_in a={};a.sin_family=AF_INET;a.sin_addr.s_addr=htonl(INADDR_LOOPBACK);
 assert(!bind(l,(sockaddr*)&a,sizeof(a)));socklen_t n=sizeof(a);assert(!getsockname(l,(sockaddr*)&a,&n));
 assert(!listen(l,1));assert(!connect(c,(sockaddr*)&a,sizeof(a)));int s=accept(l,nullptr,nullptr);assert(s>=0);
 channel ch={"TEST_UE0"};rfsimulator_state_t t={};t.role=1;t.sample_rate=122880000;
 t.buf[0]={s,100,&ch};t.nextRxTstamp=50;auto*d=rd::data(&t);
 unsigned p=rd::waiting(d,&t,&t.buf[0],100);assert(p==8);
 {rd::Scope wait(d,p);usleep(20000);}
 {rd::Scope busy(d,4);uint64_t start=rd::now(CLOCK_THREAD_CPUTIME_ID);while(rd::now(CLOCK_THREAD_CPUTIME_ID)-start<5000000){};}
 assert(d->peers[0].local_port && d->peers[0].remote_port);
 assert(d->peers[0].received_ts==100&&d->peers[0].required_ts==150);
 assert(d->counters[8].wall>=20000000 && d->counters[8].cpu<d->counters[8].wall/2);
 assert(d->counters[4].cpu>=5000000);
 uint64_t start=rd::now();for(int i=0;i<100000;i++){rd::Scope empty(d,7);}uint64_t cost=rd::now()-start;
 t.nextRxTstamp=1000;rd::progress(d,&t);assert(d->rx_ts==1000&&d->rx_calls==1);
 assert(d->counters[7].count==100000 && !(d->seq%2));
 uint64_t hist=0;for(auto x:d->counters[7].hist)hist+=x;assert(hist==100000);
 printf("{\"status\":\"PASS\",\"struct_bytes\":%zu,\"empty_scope_ns\":%.3f,\"wait_wall_ns\":%lu,\"wait_cpu_ns\":%lu,\"busy_cpu_ns\":%lu}\n",sizeof(*d),cost/100000.,d->counters[8].wall,d->counters[8].cpu,d->counters[4].cpu);
 close(s);close(c);close(l);
}
