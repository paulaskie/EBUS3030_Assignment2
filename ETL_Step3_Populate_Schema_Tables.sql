-- EBUS3030 Assignment 1 // Date: 15/08/2021 // Team5

--	Populate the BIA Inc. Star Scheme from Assignment2 ImportTable
USE Assignment2
GO

-- Removes the NULL row imported from Excel
DELETE FROM ImportTable 
WHERE Sale_Date IS NULL;

--
INSERT INTO DimDate 
	(Date_ID, Date_Month, Date_Quarter, Date_Year)
SELECT DISTINCT CAST(Sale_Date AS DATE), 
						DATEPART(MONTH, Sale_Date), 
						DATEPART(QUARTER, Sale_Date), 
						DATEPART(YEAR, Sale_Date)
FROM ImportTable

-- 
INSERT INTO DimCustomer
	(Customer_ID, Customer_First_Name, Customer_Surname)
SELECT DISTINCT Customer_ID, Customer_First_name, Customer_Surname
FROM ImportTable

-- 
INSERT INTO DimStaff
	(Staff_ID, Staff_First_Name, Staff_Surname, Staff_Office, Office_Location)
SELECT DISTINCT  Staff_ID, Staff_First_Name, Staff_Surname, Staff_Office, Office_Location
FROM ImportTable

-- 
INSERT INTO DimItem 
	(Item_ID, Item_Description, Item_Price)
SELECT DISTINCT Item_ID, Item_Description, Item_Price
FROM ImportTable



-- Populate the Fact Table
INSERT INTO FactSale
	(Receipt_Id, Receipt_Transaction_Row_ID, Sale_Date_Key, Customer_Key, Staff_Key, Item_Key, Item_Quantity, Row_Total)
	  
SELECT  x.Receipt_Id, x.Receipt_Transaction_Row_ID, d.Date_Key, c.Customer_Key, s.Staff_Key, i.Item_Key, x.Item_Quantity, x.Row_Total 
  FROM ImportTable x
	left join DimStaff s
		on x.Staff_Id = s.Staff_ID
	left join DimCustomer c
		on x.Customer_ID = c.Customer_ID
	left join DimDate d
		on x.Sale_Date = d.Date_ID
	Left join DimItem i
		on x.Item_ID = i.Item_ID
