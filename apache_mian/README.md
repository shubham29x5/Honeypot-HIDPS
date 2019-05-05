## HIDPS
copy it in `/var/log/apache2`

change IPs. Lesser IP is of HOST and Greater IP belongs to VM.

Start with forward.sh (everytime when IPs are changed)

Run flush.sh (it will flush IP Table rules after every 10mins)

loop.sh will detect script in access.log and forward the IP. (basic detection)

python file in XSS analysis will actually detect the xss using ML. (advance detection)

run backup.sh to keep the backup.
