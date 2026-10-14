"""Linux UDP source/interface binding without modifying host routing tables."""
import socket
import struct
from .identity import require


class SourceSender:
    """Unchanged datagrams, explicit IP_PKTINFO interface on every send.

    Bind(ip) alone need not select the right interface when both tunnels are
    in one host namespace. Ancillary routing metadata is local to the kernel;
    no bytes or fields are added to SFD4 or its fragment header.
    """
    def __init__(self, sock, source):
        require(sock.getsockname()[0]==source.ip,'sender socket has wrong bound lease')
        self.socket,self.source=sock,source
        self.ifindex=socket.if_nametoindex(source.tunnel)
        require(self.ifindex>0,'source tunnel disappeared')
        self.control=struct.pack('=I4s4s',self.ifindex,socket.inet_aton(source.ip),b'\0'*4)
        self.datagrams=self.bytes=0
    def sendto(self, data, destination):
        count=self.socket.sendmsg([data],[(socket.IPPROTO_IP,getattr(socket,'IP_PKTINFO',8),self.control)],0,destination)
        require(count==len(data),'partial UDP feature send')
        self.datagrams+=1; self.bytes+=count
        return count
    def __getattr__(self,name): return getattr(self.socket,name)
    def evidence(self):
        return dict(source=self.source.tag,ifindex=self.ifindex,local=self.socket.getsockname(),
                    routing='per-datagram IP_PKTINFO ifindex and source lease',datagrams=self.datagrams,bytes=self.bytes)
