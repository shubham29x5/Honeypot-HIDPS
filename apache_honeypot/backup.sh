x="$(cat access.log | grep "/admin" | tail -1 | awk '{print $7}')"
echo "$x" >> /root/Desktop/backup.txt

