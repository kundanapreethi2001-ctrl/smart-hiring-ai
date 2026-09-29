import mysql.connector

connection = mysql.connector.connect(
    host="localhost",
    user="root",
    password="admin",
    database="smart_hiring"
)

print(connection.is_connected())

cursor = connection.cursor()
cursor.execute("SELECT * FROM candidates LIMIT 5")
result = cursor.fetchone()
print(result)