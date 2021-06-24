def repeat_k_times(k):
    def decorator(train):
        def wrapper(**kwargs):
            if 'fold_no' in kwargs.keys():
                fold_no = kwargs['fold_no']
            else:
                fold_no = None
            results = []
            for i in range(k):
                best_metrics = train(train_set = kwargs['train_set'], valid_set = kwargs['valid_set'], fold_no = fold_no, run_no = i+1)
                results.append(best_metrics)
            print(f'\nAll run results : {results}')
            avg_metrics = {key:0 for key in best_metrics.keys()}
            for run in results:
                for key,val in run.items():
                    avg_metrics[key] += val
            for key in avg_metrics:
                avg_metrics[key] /= len(results)
            print(f'Averaged over runs result : {avg_metrics}')
            return avg_metrics
        return wrapper
    return decorator

