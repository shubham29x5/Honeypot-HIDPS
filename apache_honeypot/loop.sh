while :
do
	
	while IFS= read -r line; do
	    if [[ $line =~ script ]]; then
            	cat /var/www/html/adm.php > /var/www/html/admin.php

	    else
		cat /var/www/html/admi.php > /var/www/html/admin.php

            fi
	done <access.log
	cat access_reset.log > access.log
done
