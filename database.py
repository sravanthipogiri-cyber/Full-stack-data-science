import mysql.connector

conn = mysql.connector.connect(host = 'localhost',user='root',password='sravanthi@123')

if conn.is_connected():
    print('connection established')
# print(conn)

mycursor = conn.cursor()
mycursor.execute('USE internshipdb')
print(mycursor)

