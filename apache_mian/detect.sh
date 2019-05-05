while IFS= read -r line; do
    if [[ $line =~ script ]]; then
        echo "$line" >>debug.txt
	while read line; do
  		ip="$(grep -oE '[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}' <<< "$line")"
  		echo "$ip is forwarded"
		iptables -t nat -A PREROUTING -p tcp --dport 80 -j DNAT --to-destination 192.168.1.102:80
		iptables -t nat -A POSTROUTING -p tcp -d 192.168.1.102 --dport 80 -j SNAT --to-source 192.168.1.100
	done < "debug.txt"
    else
        echo "no match"
    fi
done <access.log
