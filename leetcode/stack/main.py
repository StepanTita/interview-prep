def exclusiveTime(n: int, logs: List[str]) -> List[int]:
    run = []
    res = [0] * n

    for log in logs:
        func_id, event, time = log.split(':')
        func_id = int(func_id)
        time = int(time)

        if event == 'start':
            if len(run) > 0:
                prev, prev_time = run[-1]
                res[prev] += time - prev_time

            run.append((func_id, time))
        else:
            prev, prev_time = run.pop()
            res[prev] += time - prev_time + 1

            if len(run) > 0:
                run[-1] = (run[-1][0], time + 1)
    return res