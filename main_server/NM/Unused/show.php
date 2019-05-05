<html>
<h1> Show </h1>
<?php
echo "this works"
$row = 1;
if (($handle = fopen("sample.csv","r")) !== FALSE) {
while (($data = fgetcsv($handle, 1000, ",")) !== FALSE) {
$num = count($data);
echo "
","Record #",$row,"
";
$row++;
for ($recordcount=0; $recordcount < $num; $recordcount++) {
if ($recordcount==0)
{
echo "".$data[$recordcount] . "
\n";
}
else
{
echo $data[$recordcount] . "
\n";
}
}
}
fclose($handle);
}
?>
</html>

