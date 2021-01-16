from predict_execution_review import *
from db.almaden import Sql
db = Sql("datacast2")
request_row = db.select('crawl_request','*','crawl_status="GF" and channel="navershopping"')

## pred_confing
obj_pred_config = Pred_config()

##predict
for row in request_row:
    keyword= row['keyword']
    obj_predict = Predict(obj_pred_config,keyword=keyword)
    obj_predict.predict()
    task_ids = db.select('crawl_task','*',f'keyword="{row["keyword"]}" and channel="navershopping"')
    for task in task_ids:
        db.update_one('crawl_task','crawl_status','SF','task_id',task['task_id'])
