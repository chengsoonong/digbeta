from ijcai15 import PersTour

algo = PersTour('Osaka', 'userVisits-Osak.csv')
algo.recommend('Osaka', 'rec_00.seq', 0)
#algo.recommend('Osaka', 'rec_05_time.seq', 0.5)
#algo.recommend('Osaka', 'rec_05_freq.seq', 0.5, time_based=False)
#algo.recommend('Osaka', 'rec_10_time.seq', 1.0)
#algo.recommend('Osaka', 'rec_10_freq.seq', 1.0, time_based=False)
