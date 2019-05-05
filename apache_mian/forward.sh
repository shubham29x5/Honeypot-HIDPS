#/bin/sh

echo 1 > /proc/sys/net/ipv4/ip_forward

iptables -F
iptables -t nat -F
iptables -X

iptables -t nat -A PREROUTING -p tcp --dport 80 -j DNAT --to-destination 192.168.1.114:80
iptables -t nat -A POSTROUTING -p tcp -d 192.168.1.114 --dport 80 -j SNAT --to-source 192.168.1.113
