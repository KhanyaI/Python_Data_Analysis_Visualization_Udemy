import sqlite3
import pandas as pd


con = sqlite3.connect('/Users/ifrahkhanyaree/Desktop/Kurzarbeit/Python_Pontilla_Udemy/sakila-db/sakila.db')

def sql_to_df(sql_query):
	df = pd.read_sql(sql_query,con)

	return df

query = ''' SELECT first_name, last_name FROM customer; ''' #select all from the customer table

#print(sql_to_df(query).head())

query_distinct = ''' SELECT DISTINCT(country_id) FROM city; ''' #like pandas unique
#print(sql_to_df(query_distinct).head())

query_where = ''' SELECT * FROM customer WHERE store_id = 1 ''' #filtering, like isin in pandas 

#print(sql_to_df(query_where).head())

query_and = ''' SELECT * FROM film WHERE release_year = 2006 AND rating = 'R' ''' #filtering, where both values are true
#print(sql_to_df(query_and).head())

query_or = ''' SELECT * FROM film WHERE rating = 'PG' OR rating = 'R' ''' #filtering, where one of two values is true
#print(sql_to_df(query_or).head())


query_count= ''' SELECT COUNT(customer_id) FROM customer; '''
#print(sql_to_df(query_count).head())

query_wildcard1 = ''' SELECT * FROM customer WHERE first_name LIKE 'M%'; '''
#print(sql_to_df(query_wildcard1).head())

query_wildcard2 = ''' SELECT * FROM customer WHERE last_name LIKE '_ING'; '''
#print(sql_to_df(query_wildcard2).head())


query_order = ''' SELECT * FROM customer ORDER BY last_name; ''' # sorting in pandas
#print(sql_to_df(query_order).head())


query_group = ''' SELECT store_id, COUNT(customer_id) FROM customer GROUP BY store_id; '''
print(sql_to_df(query_group).head())

