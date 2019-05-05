x="$(cat /var/log/apache2/access.log | grep "/admin" | tail -1 | awk '{print $7}')"
if echo "$x" | grep -q "%3C\|script\|alert\|"; then
	    echo "$x" >> backup.txt
	    $(bash forward.txt)
else
	    echo "no match";

fi
