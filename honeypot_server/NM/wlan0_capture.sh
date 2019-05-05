#!/bin/bash
#PATH = /root/Desktop/IITK/Live:/root/anaconda2/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
tshark -i wlan0 -w file.pcap -F libpcap -a duration:10
tshark -r file.pcap -q -z conv,ip > sample.txt
cat sample.txt | sed  's/->/ /g'|sed  's/<-/ /g'|sed  's/=/ /g'|sed  's/|/ /g'| sed  's/</ /g'| sed  's/IPv4 Conversations/ /g'|sed  's/Filter: No Filter>/ /g'|sed  's/Frames/ /g'|sed 's/Bytes/ /g'| sed  's/Duration/ /g'|sed  's/Relative Start/ /g' | cat sample.txt | sed  's/->/ /g'|sed  's/<-/ /g'|sed  's/=/ /g'|sed  's/|/ /g'| sed  's/</ /g'| sed  's/IPv4 Conversations/ /g'|sed  's/Filter: No Filter>/ /g'|sed  's/Frames/ /g'|sed 's/Bytes/ /g'| sed  's/Duration/ /g'|sed  's/Total/ /g'|cat sample.txt | sed  's/->/ /g'|sed  's/<-/ /g'|sed  's/=/ /g'|sed  's/|/ /g'| sed  's/</ /g'| sed  's/IPv4 Conversations/ /g'|sed  's/Filter: No Filter>/ /g'|sed  's/Frames/ /g'|sed 's/Bytes/ /g'| sed  's/Duration/ /g'|sed  's/Total/ /g' | sed 's/Start/ /g'|sed 's/Relative/ /g'> samplex.txt
cat samplex.txt| sed -e "s/[[:space:]]\+/ /g" | sed -e "s/ /,/g" | tail -n +6 | sed '$d' > x.txt
awk ' { if ( $1 !~ /^[ ,]/ ) { print } ; } ' x.txt > final.txt
cat final.txt | sed 's/ \+/,/g' > sample.csv
cat x.txt |awk -F "\"*,\"*" '{print $4,$6,$8,$10}'| sed 's/ \+/,/g' > band.csv
python ipvsduration.py
#cp file.pcap ./Live
python plo.py
cat sample.csv | cut -d, -f1 > baseip.txt
python3 pygeoipmap.py -o ./map.jpg -i baseip.txt
python 3v1.py
python3 pygeoipmap.py -o ./map.jpg -i baseip.txt > location.txt
cat location.txt |awk -F "\"*,\"*" '{print $1 $2}' |  sed "1,1d" | sed '$d'| sed 's/ \+/,/g' > location.csv
cat sample.csv | sed 1i"Out,In,Frames(In),Bytes(In),Frames(Out),Bytes(Out),Frames(Total),Bytes(Total),Relative_Start,Duration" > new.csv
cat band.csv | sed 1i"Out(Bytes),In(Bytes),Total(Bytes),Duration" > banded.csv


