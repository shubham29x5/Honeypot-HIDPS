import logging, sys, struct, binascii, socket
from scapy.all import *
import networkx as nx
import matplotlib.pyplot as plt

pkts=rdpcap("file.pcap",100)
def parsePcap():
    IPList = []
    for pkt in pkts:
        if pkt.haslayer(IP):
            x = pkt.getlayer(IP).src
            y = pkt.getlayer(IP).dst
            IPList.append((x, y))
    return IPList
#nodeList = set([item for pair in parseOutput for item in pair])
#print(nodeList)
parseOutput = parsePcap()
g = nx.Graph()
edgeList = parseOutput
g.add_edges_from(edgeList)
pos = nx.spring_layout(g,scale=1) #default to scale=1
nx.draw(g,pos, with_labels=True)
#plt.show()
plt.savefig('./connections.png')
