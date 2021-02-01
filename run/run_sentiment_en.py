from predict_execution_en import *
from db.almaden import Sql
db = Sql("datacast2")
key = 'sonnen home battery'
cha = 'twitter'
request_row = db.select('crawl_request','*',f'crawl_status="GF" and keyword="{key}" and channel="{cha}"')
## pred_confing

##predict
for row in request_row:
    obj_predict = Predict(keyword=row['keyword'],channel=row['channel'])
    obj_predict.predict()
    task_ids = db.select('crawl_task','*',f'keyword="{row["keyword"]}" and channel="youtube"')
    for task in task_ids:
        db.update_one('crawl_task','crawl_status','SF','task_id',task['task_id'])