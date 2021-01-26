import multiprocessing
import time
from predict_execution_blog import *
from db.almaden import Sql

#시작시간
start_time = time.time()

#멀티쓰레드 사용 하는 경우 (20만 카운트)
#Pool 사용해서 함수 실행을 병렬

if __name__ == '__main__':
    process_list = []
    task_list = []
    db = Sql("datacast2")
    task_row = db.select('''
    crawl_request AS cr JOIN crawl_request_task AS crt ON cr.request_id=crt.request_id
    JOIN request_batch AS rb ON rb.batch_id = cr.batch_id
    JOIN crawl_task AS ct ON crt.task_id=ct.task_id
    ''',
                         'rb.batch_id as batch_id,cr.request_id as request_id,ct.task_id as task_id,cr.keyword as keyword,ct.n_crawl,ct.crawl_status as crawl_status,ct.channel as channel'
                         , 'ct.crawl_status="GF" and ct.channel !="navershopping" and rb.batch_id=57 limit 6')
    ## pred_confing
    obj_pred_config = Pred_config()
    ##predict
    # #멀티 쓰레딩 Pool 사용
    for row in task_row:
        task_id = row['task_id']
        channel =row['channel']
        real_task_status = db.select('crawl_task','*','task_id=%s'%(task_id))
        if real_task_status[0]['crawl_status'] == 'GF':
            db.update_one('crawl_task', 'crawl_status', 'SI', 'task_id', task_id)
            obj_predict = Predict(obj_pred_config,task_id=task_id,keyword=row['keyword'],channel=channel)
            pool = multiprocessing.Pool(processes=6)  # 현재 시스템에서 사용 할 프로세스 개수
            pool.map(obj_predict.predict())
            pool.close()
            pool.join()

print("--- %s seconds ---" % (time.time() - start_time))
