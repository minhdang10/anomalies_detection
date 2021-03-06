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
AVG(TOTAL_AMOUNT/(BILL_WEIGHT-ACTUAL_WEIGHT+0.001)) as Avg_Dollar_Per_Pound_Diff,
MIN(LENGTH*WIDTH*HEIGHT) as Min_Dimension,
MAX(LENGTH*WIDTH*HEIGHT) as Max_Dimension,
AVG(LENGTH*WIDTH*HEIGHT) as Avg_Dimension,
MIN(TOTAL_AMOUNT/(LENGTH*WIDTH*HEIGHT+0.001)) as Min_Dollar_Per_Inch_Cube,
MAX(TOTAL_AMOUNT/(LENGTH*WIDTH*HEIGHT+0.001)) as Max_Dollar_Per_Inch_Cube,
AVG(TOTAL_AMOUNT/(LENGTH*WIDTH*HEIGHT+0.001)) as Avg_Dollar_Per_Inch_Cube

FROM "FDR_DB"."FDR_SOURCE"."DELIVERY_PACKAGE"
WHERE INDUCTION_SCAN_DATE_KEY >= '20210101';

******************************
SELECT 
MIN(INDUCT_TO_DELIVER_CALENDAR_DAYS) AS Min_Induct_To_Deliver,
MAX(INDUCT_TO_DELIVER_CALENDAR_DAYS) AS Max_Induct_To_Deliver,
AVG(INDUCT_TO_DELIVER_CALENDAR_DAYS) AS Avg_Induct_To_Deliver,
MIN(INDUCT_TO_SHIP_CALENDAR_DAYS) AS Min_Induct_To_Ship,
MAX(INDUCT_TO_SHIP_CALENDAR_DAYS) AS Max_Induct_To_Ship,
AVG(INDUCT_TO_SHIP_CALENDAR_DAYS) AS Avg_Induct_To_Ship,
MIN(SHIP_TO_USPS_CALENDAR_DAYS) AS Min_Ship_to_USPS,
MAX(SHIP_TO_USPS_CALENDAR_DAYS) AS Max_Ship_to_USPS,
AVG(SHIP_TO_USPS_CALENDAR_DAYS) AS Avg_Ship_to_USPS,
MIN(USPS_TO_DELIVER_CALENDAR_DAYS) AS Min_USPS_To_Deliver,
MAX(USPS_TO_DELIVER_CALENDAR_DAYS) AS Max_USPS_To_Deliver,
AVG(USPS_TO_DELIVER_CALENDAR_DAYS) AS Avg_USPS_To_Deliver,
MIN(INDUCT_TO_USPS_CALENDAR_DAYS) AS Min_Induct_To_USPS,
MAX(INDUCT_TO_USPS_CALENDAR_DAYS) AS Max_Induct_To_USPS,
AVG(INDUCT_TO_USPS_CALENDAR_DAYS) AS Avg_Induct_To_USPS,
MIN(SHIP_TO_DELIVER_CALENDAR_DAYS) AS Min_Ship_To_Deliver,
MAX(SHIP_TO_DELIVER_CALENDAR_DAYS) AS Min_Ship_To_Deliver,
AVG(SHIP_TO_DELIVER_CALENDAR_DAYS) AS Min_Ship_To_Deliver

FROM "FDR_DB"."FDR_SOURCE"."DELIVERY_PACKAGE"
WHERE INDUCTION_SCAN_DATE_KEY >= '20210101';

**************************
SG_Per_Day, SLA_Date, LLT....




******************************
SELECT 
MIN(TOTAL_AMOUNT-BASE_FEE_AMT) AS Min_Additional_Fee,
MAX(TOTAL_AMOUNT-BASE_FEE_AMT) AS Max_Additional_Fee,
AVG(TOTAL_AMOUNT-BASE_FEE_AMT) AS Avg_Additional_Fee

FROM "FDR_DB"."FDR_SOURCE"."DELIVERY_PACKAGE"
WHERE INDUCTION_SCAN_DATE_KEY >= '20210101';


********************************
SELECT 
MIN(AUT_DIM_LENGTH-LENGTH) AS Min_Length_Diff,
MAX(AUT_DIM_LENGTH-LENGTH) AS Max_Length_Diff,
AVG(AUT_DIM_LENGTH-LENGTH) AS Avg_Length_Diff,
MIN(AUT_DIM_WIDTH-WIDTH) AS Min_Width_Diff,
MAX(AUT_DIM_WIDTH-WIDTH) AS Max_Width_Diff,
AVG(AUT_DIM_WIDTH-WIDTH) AS Avg_Width_Diff,
MIN(AUT_DIM_HEIGHT-HEIGHT) AS Min_Height_Diff,
MAX(AUT_DIM_HEIGHT-HEIGHT) AS Max_Height_Diff,
AVG(AUT_DIM_HEIGHT-HEIGHT) AS Avg_Height_Diff

FROM "FDR_DB"."FDR_SOURCE"."DELIVERY_PACKAGE"
WHERE INDUCTION_SCAN_DATE_KEY >= '20210101';



*******************************
SELECT COUNT(*)
FROM "FDR_DB"."FDR_SOURCE"."DELIVERY_PACKAGE"
WHERE INDUCTION_SCAN_DATE_KEY >= '20210101'
AND 

(AUT_DIM_LENGTH IS NULL 
OR AUT_DIM_WIDTH IS NULL
OR AUT_DIM_HEIGHT IS NULL
OR AUT_DIM_LENGTH = 0 
OR AUT_DIM_WIDTH = 0
OR AUT_DIM_HEIGHT = 0);

--> 15867601



******************************************
SELECT 
MIN(AUT_DIM_CUBE_FILTERED-(LENGTH*WIDTH*HEIGHT)) AS Min_Dim_Diff,
MAX(AUT_DIM_CUBE_FILTERED-(LENGTH*WIDTH*HEIGHT)) AS Max_Dim_Diff,
AVG(AUT_DIM_CUBE_FILTERED-(LENGTH*WIDTH*HEIGHT)) AS Avg_Dim_Diff,
AVG(AUT_DIM_CUBE_INCH_SIZE-(LENGTH*WIDTH*HEIGHT)) AS Min_Dim_Diff_1,
AVG(AUT_DIM_CUBE_INCH_SIZE-(LENGTH*WIDTH*HEIGHT)) AS Max_Dim_Diff_1,
AVG(AUT_DIM_CUBE_INCH_SIZE-(LENGTH*WIDTH*HEIGHT)) AS Avg_Dim_Diff_1

FROM "FDR_DB"."FDR_SOURCE"."DELIVERY_PACKAGE"
WHERE INDUCTION_SCAN_DATE_KEY >= '20210101';



