import numpy as np 
import pandas as pd 

def max_exp_house_county(data):
        '''
    This function takes price paid data and returns another DataFrame containing the full details of the largest
    transaction occuring within each county present in the data
    
    '''
    #group by 'County' and for each County return row with max price (includes where more than one property has max value)
    df = data.groupby(['County',], sort=True).apply(lambda df: df.loc[df['Price'] == df['Price'].copy().max()])
    return df

    #return data.groupby('County').apply(lambda df: df.loc[df['Price'].idxmax()])

def top5_dist_qtr_val(data):
    '''
    This function will take price paid data and return a DataFrame (index by quarter) giving the 
    postcode districts(i.e AB1 2CD => AB1) with the largest total transation value for each quarter (and these values)
    
    '''
    
    
    # Convert date to datetime object and then create 'Quarter' column
    data['Date'] = pd.to_datetime(data['Date'])
    df = data[['Date','Postcode','Price']].copy()
    df['Quarter'] = df['Date'].dt.quarter
    
    
    # Convert postcode to shortened district form 
    df['Postcode'] = df['Postcode'].str.split(' ').str[0]
    
    # Create a depearate dataframe for each quarter and sort the properties by Price (descending)
    q1 = df.loc[df['Quarter'] == 1].sort_values('Price', ascending=False)[['Postcode', 'Price']].head()
    q2 = df.loc[df['Quarter'] == 2].sort_values('Price', ascending=False)[['Postcode', 'Price']].head()
    q3 = df.loc[df['Quarter'] == 3].sort_values('Price', ascending=False)[['Postcode', 'Price']].head()
    q4 = df.loc[df['Quarter'] == 4].sort_values('Price', ascending=False)[['Postcode', 'Price']].head()
    
    q1['Quarter'] = 1
    q2['Quarter'] = 2
    q3['Quarter'] = 3
    q4['Quarter'] = 4

    # Concetenate the quaters and sort by Quarter and Price
    result_df = pd.concat([q1,q2,q3,q4])
    result_df = result_df.set_index('Quarter').sort_values(['Quarter', 'Price'], ascending=[1,0])
    #return pd.pivot_table(result_df,index=['Quarter', 'Postcode' ], values='Price').sort_values('Price', ascending=[1,0])
    return result_df



def transaction_val_conc(data, perc=80):

    '''
    This function will take price paid data and return a dataframe, indexed by year and with one column for each property 
    type, giving the percentage of transactions (in descending order of size) that account for 80% of the total 
    transaction value occurring for that property type for each year.
    
    default percentage is 80, but this can be changed by adding an additional 'perc' parameter i.e perc= 70.
    
    '''
    
    data['Date'] = pd.to_datetime(data['Date'])
    temp = data.copy()
    temp['Year'] = pd.to_numeric(temp['Date'].dt.year)
    
    # get unique type and year values to iterate through
    types = temp['Type'].unique()
    years = temp['Year'].unique()
    
    # Create a new dataframe that will be returned once it caontains the information required
    new_df = pd.DataFrame(columns=['Year', 'Category', 'Concentration'])
    
    #For every year, go through every property type to calculate the concentration by category and 
    for y in years:
        for t in types:
            subset_df = temp.loc[(temp['Year'] == y) & (temp['Type'] == t)].copy()
            # Sorting values within each property type so most expensive is at the top
            subset_df.sort_values('Price', ascending=False)
            
            # Adding cumulative sum of the property prices within each property type
            subset_df['cumulative_sum'] = subset_df['Price'].cumsum()

            # Category type subset's total value
            subset_value_total = subset_df['Price'].sum()
            
            # total number of transactions for the property type
            transactions_total = len(subset_df)
            
            # calculate concentration and add row to the dataframe that will be returned - new_df
            subset_df['cumulative_pc'] = (subset_df['cumulative_sum'] / subset_value_total) * 100
            transactions_in_subset = (len(subset_df.loc[subset_df['cumulative_pc'] <= perc]))
            #print(transactions_in_subset)
            #print('\n')
            concentration = (transactions_in_subset / transactions_total)  * 100

            new_row = {'Year':y,'Category': t, 'Concentration': concentration}


            new_df = new_df.append(new_row, ignore_index = True)
    
    # Set index as year with columns for eac hcategory, values showing concentration.
    return new_df.pivot_table(index='Year', columns='Category', values='Concentration')

def compare_vol_median(first, second):
    '''
    This function will take two subsets of price paid data and returns a DataFrame showing the percentage change in the number of transactions and their median price between the two datasets, broken down by each of the following price brackets:

● £0 < x <= 250,000

● £250,000 < x <= £500,000

● £500,000 < x <= £750,000

● £750,000 < x <= £1,000,000

● £1,000,000 < x <= £2,000,000

● £2,000,000 < x <= £5,000,000

● £5,000,000+
    
    '''
    
    #index containing tuples of ranges
    ranges = [(0,250000),
                (250000, 500000),
                (500000, 750000),
                (750000, 1000000),
                (1000000, 2000000),
                (2000000,5000000),
                (5000000,)]

    # dataframe to be returned
    new_df = pd.DataFrame(index= ranges, columns=['perc_change_transactions', 'perc_change_median'])


    try:
    # Percentage change in number of transaction
        new_df.iloc[0:1,0:1] = (len(second.loc[(second['Price'] > 0) & (second['Price'] <= 250000)]) / len(first.loc[(first['Price'] > 0) & (first['Price'] <= 250000)])) *100
        new_df.iloc[1:2,0:1] = (len(second.loc[(second['Price'] > 250000) & (second['Price'] <= 500000)]) / len(first.loc[(first['Price'] > 250000) & (first['Price'] <= 500000)])) *100
        new_df.iloc[2:3,0:1] = (len(second.loc[(second['Price'] > 500000) & (second['Price'] <= 750000)]) / len(first.loc[(first['Price'] > 500000) & (first['Price'] <= 750000)])) *100
        new_df.iloc[3:4,0:1] = (len(second.loc[(second['Price'] > 750000) & (second['Price'] <= 1000000)]) / len(first.loc[(first['Price'] > 750000) & (first['Price'] <= 1000000)])) *100
        new_df.iloc[4:5,0:1] = (len(second.loc[(second['Price'] > 1000000) & (second['Price'] <= 2000000)]) / len(first.loc[(first['Price'] > 1000000) & (first['Price'] <= 2000000)])) *100
        new_df.iloc[5:6,0:1] = (len(second.loc[(second['Price'] > 2000000) & (second['Price'] <= 5000000)]) / len(first.loc[(first['Price'] > 2000000) & (first['Price'] <= 5000000)])) *100
        new_df.iloc[6:7,0:1] = (len(second.loc[second['Price'] > 5000000]) / len(first.loc[first['Price'] > 5000000])) *100
    except:
        print('Possible NaN value generated in some rows')

    try:
    # Percentage change in median price

        new_df.at[(0,250000), 'perc_change_median'] = float((second.loc[(second['Price'] > 0) & (second['Price'] <= 250000)].median() / first.loc[(first['Price'] > 0) & (first['Price'] <= 250000)].median())*100)
        new_df.at[(250000, 500000), 'perc_change_median'] = float((second.loc[(second['Price'] > 250000) & (second['Price'] <= 500000)].median() / first.loc[(first['Price'] > 250000) & (first['Price'] <= 500000)].median()) *100)
        new_df.at[(500000, 750000), 'perc_change_median'] = float((second.loc[(second['Price'] > 500000) & (second['Price'] <= 750000)].median() / first.loc[(first['Price'] > 500000) & (first['Price'] <= 750000)].median()) *100)
        new_df.at[(750000, 1000000), 'perc_change_median'] = float((second.loc[(second['Price'] > 750000) & (second['Price'] <= 1000000)].median() / first.loc[(first['Price'] > 750000) & (first['Price'] <= 1000000)].median()) *100)
        new_df.at[(1000000, 2000000), 'perc_change_median'] = float((second.loc[(second['Price'] > 1000000) & (second['Price'] <= 2000000)].median() / first.loc[(first['Price'] > 1000000) & (first['Price'] <= 2000000)].median()) *100)
        new_df.at[(2000000,5000000), 'perc_change_median'] = float((second.loc[(second['Price'] > 2000000) & (second['Price'] <= 5000000)].median() / first.loc[(first['Price'] > 2000000) & (first['Price'] <= 5000000)].median()) *100)
        new_df.at[(5000000,), 'perc_change_median'] = float(second.loc[second['Price'] > 5000000].median() / first.loc[first['Price'] > 5000000].median() *100)
    except:
        print('Possible NaN value generated in some rows')
    return new_df




# columns=['address', 'holding_periods', 'appearances', 'year', 'annualised_return']

def avg_annualised_return(input_df, output_log=False):
'''
This function takes price paid data and returns the
average length of a holding period and the annualised change in value between the purchase
and sale, grouped by the year a holding period ends and the property type.
''' 

    from dateutil import relativedelta
    # Creating a separate field for that combined the house number, flat number 
    # and postcode to ensure multiple properties with the same postcode are treated as separate.
        
    data = input_df.copy()
    data['Date'] = pd.to_datetime(data['Date'])
    data['Year'] = data['Date'].dt.year

    data['num_postcode'] = data['PAON'] + data['SAON'].astype(str) + data['Postcode']
    data['num_postcode'] = data['num_postcode'].astype(str)
    data = data.sort_values('Date', ascending=True)
    working_df = pd.DataFrame(columns=['year', 'type', 'hold_period', 'annualised_return']) 

    # For each
    for address in data['num_postcode']:
        current = data.loc[data['num_postcode'] == address]
        appearances = len(current)
        
        # don't add to new dataset if there is no holding period

        if appearances < 2:
            
            continue
        
        # annualized_return=((1 + total_return)**(months))-1
        if appearances > 1:
            
            for period in range(appearances-1):

                if period == appearances-1:
                    break
                    
                new_type = current['Type'].iloc[period]
                
                total_return = (current['Price'].iloc[period+1] - current['Price'].iloc[period]) / current['Price'].iloc[period]
                
                # calculation plus 1 to avoid divide by zero error. For properties sold in the first month (0), it is treated as month = 1
                hold_period = float(relativedelta.relativedelta(current['Date'].iloc[period+1], current['Date'].iloc[period]).months)+1
                
                # year of the sale (next period)
                year = current['Year'].iloc[period+1]
                
                ann_return = (((1 + total_return)**(12/hold_period))-1)*100
                new_row = {'year': year, 'type': new_type, 'hold_period': hold_period, 'annualised_return': ann_return}
                
                working_df = working_df.append(new_row, ignore_index=True)
                
                
                # I used this for checking / debugging
                if output_log == True:
                    print('---New holding period ---\n')
                    print(new_row)
                    print(hold_period)
                    print('Sale Price: ' + str(current['Price'].iloc[period+1]) + ' Buy Price: ' + str(current['Price'].iloc[period]) + '\n\n')

    return working_df.set_index(['year', 'type']).groupby(level=[0,1]).mean()