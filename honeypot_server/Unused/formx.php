<?php
$servername = "localhost";
$username = "saubhagya";
$password = "root";
$dbname = "IITK";
$conn = new mysqli($servername, $username, $password, $dbname);
// Check connection
if ($conn->connect_error) {
    die("Connection failed: " . $conn->connect_error);
}

$sql = "INSERT into MyGuests (lastname) VALUES ('".$_POST["lname"]."')";

if ($conn->query($sql) === TRUE) {
    echo "Record updated successfully";
} else {
    echo "Error updating record: " . $conn->error;
}

$conn->close();
?>

