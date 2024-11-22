from ml_collections import ConfigDict
PoCConfig = ConfigDict()
PoCConfig.BASE_URL = "http://dev.jinwoosi.co.kr:8083/v1/{target}?start_date_time={start_time}&end_date_time={end_time}&page_number=1&number_of_rows=20&lang=ko"