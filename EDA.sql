SELECT COUNT(*)
FROM  "FDR_NGSDW_DB_PROD"."FDRDW"."FWDPACKAGERATING"
WHERE CREATEDATEKEY >= '20210101';

SELECT *
FROM  "FDR_NGSDW_DB_PROD"."FDRDW"."FWDPACKAGERATING"
WHERE CREATEDATEKEY >= '20210101'
LIMIT 20;

What needs to be examined
feature of the package 
delivery pakcage table 


******
SELECT COUNT(*) AS Number_of_Records,
//COUNT(DISTINCT MERCHANTORGKEY) AS Number_of_Merchant_Org_Key,
MIN(ACTUALWEIGHT) as Min_Act_Weight,
MAX(ACTUALWEIGHT) as Max_Act_Weight,
AVG(ACTUALWEIGHT) as Avg_Act_Weight,
MIN(LENGTH) as Min_Length,
MAX(LENGTH) as Max_Length,
AVG(LENGTH) as Avg_Length,
MIN(WIDTH) as Min_Width,
MAX(WIDTH) as Max_Width,
AVG(WIDTH) as Avg_Width,
MIN(HEIGHT) as Min_Height,
MAX(HEIGHT) as Max_Height,
AVG(HEIGHT) as Avg_Height,
MIN(LENGTH*WIDTH*HEIGHT) as Min_Dimension,
MAX(LENGTH*WIDTH*HEIGHT) as Max_Dimension,
AVG(LENGTH*WIDTH*HEIGHT) as Avg_Dimension,
MIN(BILLINGWEIGHT) as Min_Billing_Weight,
MAX(BILLINGWEIGHT) as Max_Billing_Weight,
AVG(BILLINGWEIGHT) as Max_Billing_Weight,
MIN(BILLINGWEIGHT-ACTUALWEIGHT) as Min_Weight_Diff,
MAX(BILLINGWEIGHT-ACTUALWEIGHT) as Max_Weight_Diff,
AVG(BILLINGWEIGHT-ACTUALWEIGHT) as Avg_Weight_Diff,
MIN(TOTALFEEAMT) as Min_Total_Fee,
MAX(TOTALFEEAMT) as Max_Total_Fee,
AVG(TOTALFEEAMT) as Avg_Total_Fee,
MIN(TOTALFEEAMT/BILLINGWEIGHT) as Min_Dollar_Per_Pound,
MAX(TOTALFEEAMT/BILLINGWEIGHT) as Max_Dollar_Per_Pound,
AVG(TOTALFEEAMT/BILLINGWEIGHT) as Avg_Dollar_Per_Pound,
MIN(TOTALFEEAMT/(LENGTH*WIDTH*HEIGHT+0.001)) as Min_Dollar_Per_Inch_Cube,
MAX(TOTALFEEAMT/(LENGTH*WIDTH*HEIGHT+0.001)) as Max_Dollar_Per_Inch_Cube,
AVG(TOTALFEEAMT/(LENGTH*WIDTH*HEIGHT+0.001)) as Avg_Dollar_Per_Inch_Cube,
MIN(TOTALFEEAMT/(BILLINGWEIGHT-ACTUALWEIGHT+0.001)) as Min_Dollar_Per_Pound_Diff,
MAX(TOTALFEEAMT/(BILLINGWEIGHT-ACTUALWEIGHT+0.001)) as Max_Dollar_Per_Pound_Diff,
AVG(TOTALFEEAMT/(BILLINGWEIGHT-ACTUALWEIGHT+0.001)) as Avg_Dollar_Per_Pound_Diff

FROM "FDR_NGSDW_DB_PROD"."FDRDW"."FWDPACKAGERATING"
WHERE CREATEDATEKEY >= '20210101';



********************
SELECT COUNT(*) AS Number_of_Records
FROM "FDR_NGSDW_DB_PROD"."FDRDW"."FWDPACKAGERATING"
WHERE CREATEDATEKEY >= '20210101'
AND (LENGTH = 0 
OR WIDTH = 0
OR HEIGHT = 0);



--> 11351793

***********************
SELECT COUNT(*) AS Number_of_Records
FROM "FDR_NGSDW_DB_PROD"."FDRDW"."FWDPACKAGERATING"
WHERE CREATEDATEKEY >= '20210101'
AND 

(LENGTH IS NULL 
OR WIDTH IS NULL
OR HEIGHT IS NULL)

OR/AND

(LENGTH = 0 
OR WIDTH = 0
OR HEIGHT = 0);

--> 47927847 / 49

***************
SELECT COUNT(*) AS Number_of_Records
FROM "FDR_NGSDW_DB_PROD"."FDRDW"."FWDPACKAGERATING"
WHERE CREATEDATEKEY >= '20210101'
AND 

(LENGTH IS NULL 
OR WIDTH IS NULL
OR HEIGHT IS NULL);

---> 14046091

*********************
SELECT COUNT(*) AS Number_of_Records
FROM "FDR_NGSDW_DB_PROD"."FDRDW"."FWDPACKAGERATING"
WHERE CREATEDATEKEY >= '20210101'
AND 

(LENGTH IS NULL 
OR WIDTH IS NULL
OR HEIGHT IS NULL
OR LENGTH = 0 
OR WIDTH = 0
OR HEIGHT = 0);

--> 25397835


feel, understanding, value 






*************************
*******
SELECT *
FROM "FDR_NGSDW_DB_PROD"."FDRDW"."PACKAGEWEIGHTDIM"
WHERE FACILITYLOCALCREATEDATEKEY >= '20210101'
LIMIT 20;

**********

SELECT COUNT(*)
FROM "FDR_NGSDW_DB_PROD"."FDRDW"."PACKAGEWEIGHTDIM"
WHERE FACILITYLOCALCREATEDATEKEY >= '20210101';

--> 86128126



********************
SELECT COUNT(*) AS Number_of_Records,
MIN(WEIGHT) as Min_Weight,
MAX(WEIGHT) as Max_Weight,
AVG(WEIGHT) as Avg_Weight,
MIN(LENGTH) as Min_Length,
MAX(LENGTH) as Max_Length,
AVG(LENGTH) as Avg_Length,
MIN(WIDTH) as Min_Width,
MAX(WIDTH) as Max_Width,
AVG(WIDTH) as Avg_Width,
MIN(HEIGHT) as Min_Height,
MAX(HEIGHT) as Max_Height,
AVG(HEIGHT) as Avg_Height
FROM "FDR_NGSDW_DB_PROD"."FDRDW"."PACKAGEWEIGHTDIM"
WHERE FACILITYLOCALCREATEDATEKEY >= '20210101';




*************************
SELECT * 
FROM "FDR_DB"."FDR_SOURCE"."DELIVERY_PACKAGE"
WHERE INDUCTION_SCAN_DATE_KEY >= '20210101'
LIMIT 20;

****
SELECT COUNT(*)
FROM "FDR_DB"."FDR_SOURCE"."DELIVERY_PACKAGE"
WHERE INDUCTION_SCAN_DATE_KEY >= '20210101';

--> 65957044


******************
SELECT COUNT(*) 
FROM "FDR_DB"."FDR_SOURCE"."DELIVERY_PACKAGE"
WHERE INDUCTION_SCAN_DATE_KEY >= '20210101'
AND 

(LENGTH IS NULL 
OR WIDTH IS NULL
OR HEIGHT IS NULL
OR LENGTH = 0 
OR WIDTH = 0
OR HEIGHT = 0);


--> 25490774


**********
SELECT COUNT(*) AS Number_of_Records,
MIN(LENGTH) as Min_Length,
MAX(LENGTH) as Max_Length,
AVG(LENGTH) as Avg_Length,
MIN(WIDTH) as Min_Width,
MAX(WIDTH) as Max_Width,
AVG(WIDTH) as Avg_Width,
MIN(HEIGHT) as Min_Height,
MAX(HEIGHT) as Max_Height,
AVG(HEIGHT) as Avg_Height,
MIN(ACTUAL_WEIGHT) as Min_Act_Weight,
MAX(ACTUAL_WEIGHT) as Max_Act_Weight,
AVG(ACTUAL_WEIGHT) as Avg_Act_Weight,
MIN(BILL_WEIGHT) as Min_Bill_Weight,
MAX(BILL_WEIGHT) as Max_Bill_Weight,
AVG(BILL_WEIGHT) as Avg_Bill_Weight,
MIN(BILL_WEIGHT-ACTUAL_WEIGHT) as Min_Weight_Diff,
MAX(BILL_WEIGHT-ACTUAL_WEIGHT) as Max_Weight_Diff,
AVG(BILL_WEIGHT-ACTUAL_WEIGHT) as Avg_Weight_Diff,
MIN(TOTAL_AMOUNT) as Min_Total_Fee,
MAX(TOTAL_AMOUNT) as Max_Total_Fee,
AVG(TOTAL_AMOUNT) as Avg_Total_Fee,
MIN(TOTAL_AMOUNT/BILL_WEIGHT) as Min_Dollar_Per_Pound,
MAX(TOTAL_AMOUNT/BILL_WEIGHT) as Max_Dollar_Per_Pound,
AVG(TOTAL_AMOUNT/BILL_WEIGHT) as Avg_Dollar_Per_Pound,
MIN(TOTAL_AMOUNT/(BILL_WEIGHT-ACTUAL_WEIGHT+0.001)) as Min_Dollar_Per_Pound_Diff,
MAX(TOTAL_AMOUNT/(BILL_WEIGHT-ACTUAL_WEIGHT+0.001)) as Max_Dollar_Per_Pound_Diff,
AVG(TOTAL_AMOUNT/(BILL_WEIGHT-ACTUAL_WEIGHT+0.001)) as Avg_Dollar_Per_Pound_Diff

FROM "FDR_DB"."FDR_SOURCE"."DELIVERY_PACKAGE"
WHERE INDUCTION_SCAN_DATE_KEY >= '20210101';