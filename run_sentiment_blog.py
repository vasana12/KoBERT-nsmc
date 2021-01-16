from predict_execution_blog import *
from db.almaden import Sql
db = Sql("datacast2")
request_row = db.select('crawl_request','*','crawl_status="GF" and channel="naverblog" group by keyword')
## pred_confing
obj_pred_config = Pred_config()

##predict
for row in request_row:
    obj_predict = Predict(obj_pred_config,keyword=row['keyword'])
    obj_predict.predict()
    task_ids = db.select('crawl_task','*',f'keyword="{row["keyword"]}" and channel="naverblog"')
    for task in task_ids:
        db.update_one('crawl_task','crawl_status','SF','task_id',task['task_id'])