import os
import csv
import glob 

PATH_LOGS = "logs/lensnerf"
# EXP_NAME = "wacv23_exp00007.3"
#EXP_NAME = "wacv23_wacv0617"
EXP_NAME = "wacv23_wacv0620_basemodel"

final_metric_dir = os.path.join('logs', 'metrics')
os.makedirs(final_metric_dir, exist_ok=True)
fpath = os.path.join(final_metric_dir, EXP_NAME+'.csv')

#todo = ['AmusementPark', 'Gink', 'Sheep']
todo = []
with open(fpath, 'w') as f:
    
    agg_title_longest =[]
    agg_all = []
    # for each log
    for dir in os.listdir(PATH_LOGS):
        if not (EXP_NAME in dir):
            continue

        # for each render test
        for render_test in sorted(os.listdir(os.path.join(PATH_LOGS, dir)), reverse=True):
            if not ('render_test' in render_test):
                continue
                
            # for each csv, parse exp / data / fnumber / detail info
            agg = []
            dataF, details = dir.split(EXP_NAME)
            data, _, ftrain, _, ftest, _ = dataF.split('_')

            if len(todo) and ( not (data in todo) ):
                continue
            print(f'data name: {dataF}')

            agg.append(EXP_NAME)
            agg.append(data)
            agg.append(ftrain)
            agg.append(ftest)
            agg.append(details)
            

            # file_metric = os.path.join(PATH_LOGS, dir, render_test, 'metric_avr.csv')
            file_metrics = os.path.join(PATH_LOGS, dir, render_test, '*metric_avr.csv')
            file_metrics = sorted(glob.glob(file_metrics))

            if len(file_metrics) == 0:
                print(f'no such file *metric_avr.csv')
                break
            else:
                file_metric = file_metrics[-1]
                                
            if not os.path.exists(file_metric):
                print(f'no such file {file_metric}')
                break
            with open(file_metric) as csvfile:

#            with open(os.path.join(PATH_LOGS, dir, render_test, 'metric_avr.csv')) as csvfile:
                contents = csv.reader(csvfile)
                
                for row in contents :
                    if len(row) != 5:
                        continue
                    agg.append(row[0]) # time or index
                    agg.append(row[1]) # psnr
                    agg.append(row[2]) # ssim
                    agg.append(row[3]) # lpips
                    agg.append(row[4]) # dists
            
            # make title ----------------------------------- 
            agg_title = []  
            # default 10
            agg_title.append('exp')
            agg_title.append('data')
            agg_title.append('Ftrain')
            agg_title.append('Ftest')
            agg_title.append('details')
            agg_title.append('time(avr)(s)')
            agg_title.append('PSNR(avr)')
            agg_title.append('SSIM(avr)')
            agg_title.append('LPIPS(avr)')
            agg_title.append('DISTS(avr)')
            
            for i in range((len(agg)-10)//5):
                agg_title.append('ImgID')
                agg_title.append('PSNR')
                agg_title.append('SSIM')
                agg_title.append('LPIPS')
                agg_title.append('DISTS')                
            
            if len(agg_title ) > len(agg_title_longest):
                agg_title_longest = agg_title
                #print(len(agg_title_longest), data, ftrain, ftest)
            agg_all.append(agg)
            # write title & contents  -----------------------------------

            break # handle last test dir only
    f.write(','.join(agg_title_longest) + '\n')
    for agg in agg_all:
        #print(len(agg))
        f.write(','.join(agg) + '\n')
