new customer flag , first time order palcer

feature that capture number of oreders returned before

place special emphasis on feature engineering 


search how to deal with a a lot fo categories 

do a stacked barplot for all the catgories 
in two categories , retruned / not returned

and see if any minority categories 




target encoder for customer name feature

that captures information for customer 





the incoming data is clean 
there are no null values 
there are no rogue/incorrect values to 
	( as seen in the df.describe and thhe value_counts of the categroical varaibles )


there are no issues in the temporal columns of Order.Date , Ship.Date 
no extreme values



the level/grain of the data is at Order-ID , 
there are no rows that violate this level



----

since customer_name is a categorical variable with ~800 uynique values , 
the ebst way to treat it would be to do target_mean_encoding to rate of the target variable of return_status 


ie if John Doe return 4 of 5 orders in his history .
then the vlaue will be 0.8

most customers who dont return anything will have this feaature as 0 

new customers will also have this as 0




similiary instead of jsut a fraction , we can also generate a time series feature as 
of number fo returns uptil date 

this feature has to be implemented as a time series 


this will help us 

that too will be 0 for any new customer whio has not been recorded so far 




the main focus is on creating a pieline that can take in data as a CSV dump in the given format and train itself 

and during deployment also be able to handle neew incoming data in the same formart



hence the features should not include anything that is not going to be available in the future 
( for eaxmple date )
and also 











pd.lag for pandas 





investigate cancellation first , is it growing or dropping