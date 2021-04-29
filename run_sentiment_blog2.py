from predict_execution_blog import *
from db.almaden import Sql
db = Sql("datacast2")

blog_channel_list = str(('naverblog','instagram','GooglePlay','forum','navercafe'))
review_channel_list = str(('navershopping','youtube'))

task_row = db.select('''
crawl_request AS cr JOIN crawl_request_task AS crt ON cr.request_id=crt.request_id
JOIN request_batch AS rb ON rb.batch_id = cr.batch_id
JOIN crawl_task AS ct ON crt.task_id=ct.task_id
''','rb.batch_id as batch_id,cr.request_id as request_id,ct.task_id as task_id,cr.keyword as keyword,ct.n_crawl,ct.crawl_status as crawl_status,ct.channel as channel'
,'ct.crawl_status="GF" and rb.batch_id=%s and ct.channel in %s'%(96,blog_channel_list))
## pred_confing
obj_pred_config = Pred_config()

##predict
for row in task_row:

    task_id = row['task_id']
    channel =row['channel']

    real_task = db.select('crawl_task','*','task_id=%s'%(task_id))
    real_crawl_status = real_task[0]['crawl_status']
    print(task_id)
    if real_crawl_status=='GF':
        db.update_one('crawl_task', 'crawl_status', 'SI', 'task_id', task_id)
        obj_predict = Predict(obj_pred_config,task_id=task_id,keyword=row['keyword'],channel=row['channel'])
        obj_predict.predict()
        db.update_one('crawl_task','crawl_status','SF','task_id',task_id)
