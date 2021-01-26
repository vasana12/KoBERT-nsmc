from predict_execution_blog_groupby_request import *
from db.almaden import Sql
db = Sql("datacast2")

task_row = db.select(
'''
crawl_request AS cr 
JOIN crawl_request_task as crt on cr.request_id=crt.request_id 
join crawl_task as ct on ct.task_id = crt.task_id
join request_batch AS rb ON rb.batch_id = cr.batch_id
''',
'rb.batch_id as batch_id,cr.request_id as request_id,ct.task_id as task_id, cr.keyword as keyword,cr.channel as channel',
'cr.crawl_status="GF" and cr.channel !="navershopping" and rb.batch_id=57'
,asDataFrame=True)
request_row = task_row.drop_duplicates(subset='request_id')
## pred_confing
obj_pred_config = Pred_config()

##predict
for idx in request_row.index:
    request_id = request_row.at[idx,'request_id']
    keyword = request_row.at[idx,'keyword']
    channel = request_row.at[idx, 'channel']
    task_row = task_row[task_row['request_id']==request_id]

    for idx2 in task_row.index:
        task_id = task_row.at[idx,'task_id']
        db.update_one('crawl_task', 'crawl_status', 'SI', 'task_id', task_id)
    obj_predict = Predict(obj_pred_config,request_id=request_id,keyword=keyword,channel=channel)
    obj_predict.predict()

    for idx3 in task_row.index:
        task_id = task_row.at[idx,'task_id']
        db.update_one('crawl_task', 'crawl_status', 'SF', 'task_id', task_id)
