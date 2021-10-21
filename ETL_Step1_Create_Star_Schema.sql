-- EBUS3030 Assignment 1 // Date: 14/08/2021 // Team 5


-- Deletes database is exists
USE MASTER
GO
IF DB_ID (N'Assignment2') IS NOT NULL -- Check if it already exists and drop if it is
DROP DATABASE Assignment2;
GO


-- Create Database
CREATE DATABASE Assignment2;
GO

-- Selects Database
USE Assignment2
GO

-- Creates a Star Schema for Assignment 1

-- Drop tables if they exist
DROP TABLE IF EXISTS FactSale;							-- Need to drop FactTable before dropping other tables
DROP TABLE IF EXISTS DimItem;							-- After FactTable has been dropped so it is safe to drop Item Table
DROP TABLE IF EXISTS DimDate;
DROP TABLE IF EXISTS DimCustomer;
DROP TABLE IF EXISTS DimStaff;

GO

-- Creates empty table for items
CREATE TABLE DimItem (					
	Item_Key int identity not null,						-- Auto-incremented table key
	Item_ID int not null,						        -- Item key from excel file
	Item_Description nvarchar(30) null,
	Item_Price money null,
	
 PRIMARY KEY (Item_Key)
 )

-- Creates empty table for dates, Date_Month, Date_Quarter, Date_Year will be populated using DATEPART function
CREATE TABLE DimDate (
	Date_Key int identity not null,
	Date_ID date not null,								-- This will the actual date
	Date_Month int null,								-- Calculated Month from date using DATEPART
	Date_Quarter int null,								-- Calculated Quarter from date using DATEPART
	Date_Year int null,									-- Calculated Year from date using DATEPART

 PRIMARY KEY (Date_Key)
 )

-- Creates empty table for customers
CREATE TABLE DimCustomer (
	Customer_Key int identity not null,					-- Auto-incremented table key
	Customer_ID nvarchar(5) not null,					-- Customer key from excel file
	Customer_First_Name nvarchar(255) null,
	Customer_Surname nvarchar(255) null,					
 
 PRIMARY KEY (Customer_Key)
 )

-- Creates empty table for staff
  CREATE TABLE DimStaff (
	Staff_Key int identity not null,					-- Auto-incremented table key	
	Staff_ID nvarchar(5) not null,						-- Staff key from excel file
	Staff_First_Name nvarchar(20) null,				
	Staff_Surname nvarchar(20) null,
	Staff_Office int null,       -- Could be split off into another table. For simplicity, as there is only 1 office location, we left the non-normalised association
	Office_Location nvarchar(50) -- Could be split off into another table. For simplicity, as there is only 1 office location, we left the non-normalised association

 PRIMARY KEY (Staff_Key)
	)

-- Create the Fact Table (central table of the Star Diagram) for each Sale transaction

CREATE TABLE FactSale (
	Sale_Key int identity not null,						-- Auto-incremented table key
	Receipt_ID int null,								-- Receipt key from excel file
	Receipt_Transaction_Row_ID int null,				-- Line number of receipt from excel file
	Sale_Date_Key int null,								
	Customer_Key int null,
	Staff_Key int null,
	Item_Key int null,
	Item_Quantity int null,
	Row_Total money null,

-- links the Dimension tables to the Fact tables	
 FOREIGN KEY (Item_Key) REFERENCES DimItem (Item_Key),
 FOREIGN KEY (Sale_Date_Key)  REFERENCES DimDate (Date_Key),
 FOREIGN KEY (Customer_Key) REFERENCES DimCustomer (Customer_Key),
 FOREIGN KEY (Staff_Key) REFERENCES DimStaff (Staff_Key)
 )							

 GO