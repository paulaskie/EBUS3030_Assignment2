-- EBUS3030 Assignment 1 // Date: 15/08/2021 // Team 5

DROP TABLE IF EXISTS ImportTable;

USE Assignment2
GO

/****** Object:  Table: Assignment2.ImportTable    Script Date: 14/08/2021  ******/
SET ANSI_NULLS ON
GO

SET QUOTED_IDENTIFIER ON
GO

DROP TABLE IF EXISTS ImportTable;

CREATE TABLE ImportTable(
	Sale_Date datetime null,
	Receipt_ID int null,
	Customer_ID nvarchar(5) null,
	Customer_First_Name nvarchar(255) null,
	Customer_Surname nvarchar(255) null,
	Staff_ID nvarchar(5) null,
	Staff_First_Name nvarchar(255) null,
	Staff_Surname nvarchar(255) null,
	Staff_Office int null,
	Office_Location nvarchar(255) null,
	Receipt_Transaction_Row_ID int null,
	Item_ID int null,
	Item_Description nvarchar(255) null,
	Item_Quantity int null,
	Item_Price money null,
	Row_Total money null
) ON [PRIMARY]								--signifies the primary storage
GO


-- Steps to Import Excel File are as follows:

-- 64-bit System:

--	1. Open SQL Server 2019 Import and Export Data (64-bit) then click next on the start page.
--	2. On the "Choose Data Source" page select the dropdown box and select "Microsoft Excel" browse to the file all options should be correct for the sample file
--	3. On the "Choose a Destination" page select the dropdown box and select "Microsoft OLE DB Driver for SQL Server" then select the database that you will be writing to
--	   In this case it will be "Assignment2" then click next.
--	4. On the "Specify Table Copy or Query" page select "Copy data from one or more tables or views", this should already be selected. Click next.
--	5. On the "Select Source Tables and Views" shows the sheets in the excel file. Select the sheet containing the table should look like "Asgn1 Data$" then in the 
--     "Destination: (local)" column clicck on the value next to the sheet name and select "[dbo].[ImportTable]". Click "Edit Mappings" to make sure input data is being
--	   mapped to the correct. If the mappings are incorrect, map the correct input to the correct database target. When correct OK then click Next.
--	6. On the "Review Data Type Mapping" you can select "Ignore" on the "On Error (global)" and "On Trucation (global)" then select Next 
--	7. On the "Save and Run Package" "Run immediatley" should already be select, click Next.
--	8. On the "Complete the Wizard" page click Finish.
--  9. If successful the title will change to "The execution was successful". You can then close the wizard.

/*
SSMS is available only as a 32-bit application for Windows. If you need a tool that runs on operating systems other than Windows, 
we recommend Azure Data Studio. Azure Data Studio is a cross-platform tool that runs on macOS, Linux, as well as Windows. 
For details, see Azure Data Studio.

Need to run SQL Server Import and Export (64-Bit) to import the data from a 64-Bit system. 
The SQL Server Import and Export in tasks is 32-Bit and errors when run.
*/