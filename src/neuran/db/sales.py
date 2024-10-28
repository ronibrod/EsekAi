import calendar
from .index import get_sales_collection

def get_first_and_last_sale_dates(user_name):
    sales_collection = get_sales_collection(user_name)
    first_sale = sales_collection.find_one({}, sort=[('date', 1)])  # Ascending order for the first sale
    last_sale = sales_collection.find_one({}, sort=[('date', -1)])  # Descending order for the last sale
    
    if first_sale and last_sale:
        return first_sale['date'], last_sale['date']
    else:
        return None, None

def get_list_of_sales(user_name, product=[]):
    sales_collection = get_sales_collection(user_name)
    first_date, last_date = get_first_and_last_sale_dates(user_name)
    
    if not first_date or not last_date:
        return []
    
    query = {
      'date': {'$gte': first_date, '$lte': last_date}
    }
    if len(product) > 0:
        query['product'] = product['name']
        
    return list(sales_collection.find(query))

def get_sorted_sale_data(list_of_sales, first_date, last_date, time_division='hourly'):
    first_year = first_date.year
    last_year = last_date.year - first_year
    first_month = first_date.month - 1
    last_month = last_date.month
    first_day_of_month = first_date.day - 1
    last_day_of_month = last_date.day
    
    sorted_sales = []
    for year in range(0, last_year + 1):
        sorted_sales.append([])
        
        start_month = 0 if year != 0 else first_month
        end_month = 12 if year != last_year else last_month
        for month in range(start_month, end_month):
            
            sorted_sales[year].append([])
            
            start_day_of_month = 0 if year != 0 and month != first_month else first_day_of_month
            end_day_of_month = calendar.monthrange(year + first_year, month + 1)[1] if not (year == last_year and month == last_month) else last_day_of_month
            for day in range(start_day_of_month, end_day_of_month):
                sorted_sales[year][month].append([])
                
                for hour in range(0, 24):
                    sorted_sales[year][month][day].append([])
    
    for sale in list_of_sales:
        year = sale['year']
        month = sale['month'] - 1
        day_of_month = sale['day_of_month'] - 1
        hour = sale['hour']
        
        sorted_sales[year - first_year][month][day_of_month][hour].append(sale)
    
    
    
    # first_year = list_of_sales[0]['year']

    # sorted_sales = []
    # for sale in list_of_sales:
    #     year = sale['year']
    #     month = sale['month'] - 1
    #     day_of_month = sale['day_of_month'] - 1
    #     hour = sale['hour']
        
    #     while len(sorted_sales) <= year - first_year:
    #         sorted_sales.append([])
    #     while len(sorted_sales[year - first_year]) <= month:
    #         sorted_sales[year - first_year].append([])
    #     while len(sorted_sales[year - first_year][month]) <= day_of_month:
    #         sorted_sales[year - first_year][month].append([])
    #     while len(sorted_sales[year - first_year][month][day_of_month]) < 24:
    #         sorted_sales[year - first_year][month][day_of_month].append([])

    #     sorted_sales[year - first_year][month][day_of_month][hour].append(sale)
        
    sorted_sale_data_by_sequence = []
    sorted_sale_data = []
    for year in sorted_sales:
        for month in year:
            for day in month:
                for hour in day:
                    for sale in hour:
                        sorted_sale_data.append(sale)
                    if time_division == 'hourly':
                        sorted_sale_data_by_sequence.append(sorted_sale_data)
                        sorted_sale_data = []
                if time_division == 'daily':
                    sorted_sale_data_by_sequence.append(sorted_sale_data)
                    sorted_sale_data = []
            if time_division == 'monthly':
                sorted_sale_data_by_sequence.append(sorted_sale_data)
                sorted_sale_data = []
        if time_division == 'yearly':
            sorted_sale_data_by_sequence.append(sorted_sale_data)
            sorted_sale_data = []
    
    # while sorted_sale_data_by_sequence and len(sorted_sale_data_by_sequence[0]) == 0:
    #     sorted_sale_data_by_sequence.pop(0)

    return sorted_sale_data_by_sequence
