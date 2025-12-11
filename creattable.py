import mysql.connector

conn = mysql.connector.connect(host = 'localhost',user='root',password='sravanthi@123',database='internshipdb')

mycursor = conn.cursor()
mycursor.execute('create table student(name varchar(50),branch varchar(10), id int)')
mycursor.execute('show tables')

for x in mycursor:
    print(x)