import sys
from pyspark import SparkConf, SparkContext
from math import sqrt
import numpy as np

sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf8', buffering=1)

def compMovGen(movie_pair):
    global avg_age_dicts
    global avg_age_dicts2
    # Compute genre similarity
    similarMovieID = movie_pair[0][0]
    movieID = movie_pair[0][1]
    a = genDict[similarMovieID]
    b = genDict[movieID]
    c= 0
    d= 0
    for x, y in zip(a, b):
        if x == y:
            c+= 1
        
    res = c/float(19)
    if (b[15]=='1' and a[15]=='1'):
        res = res+0.1
    if (b[8]=='1' and a[8]=='1'):
        res = res+0.1
    if (b[3]=='1' and a[3]=='1'):
        res = res+0.1
    if (b[4]=='1' and a[4]=='1'):
        res = res+0.1
    if (b[6]=='1' and a[6]=='1') and (b[16]=='1' and a[16]=='1'):
        res = res+0.1
    if (b[1]=='1' and a[1]=='1') and (b[11]=='1' and a[11]=='1'):
        res = res+0.1
    if (b[12]=='1' and a[12]=='1') and (b[14]=='1' and a[14]=='1'):
        res = res+0.1
    if (b[8]=='1' and a[8]=='1') and (b[14]=='1' and a[14]=='1'):
        res = res+0.1
    if b[11]=='1' and a[11]=='1':
        res = res+0.1
    if b[18]=='1' and a[18]=='1':
        res = res+0.1
    
    # Compute age similarity
    #age1 = avg_age_dicts[similarMovieID]
    #age2 = avg_age_dicts[movieID]
    #std1 = avg_age_dicts2[similarMovieID]
    #std2 = avg_age_dicts2[movieID]

    #ag_res = 0.1*(abs(age1-age2)) #+0.05*abs(std1-std2)
    #res = res-ag_res
    # Movie score (cossine similarity)
    numPairs = 0
    sum_xx = sum_yy = sum_xy = 0
    for ratingX, ratingY in movie_pair[1]:
        sum_xx += ratingX * ratingX
        sum_yy += ratingY * ratingY
        sum_xy += ratingX * ratingY
        numPairs += 1

    numerator = sum_xy
    denominator = sqrt(sum_xx) * sqrt(sum_yy)

    score = 0
    if (denominator):
        score = (numerator / (float(denominator)))
    score = round(score,3)
    
    return (round(res,3),score)
    

def Us_age():
    us__ages = {}
    with open("ml-100k/u.user", encoding='ascii', errors='ignore') as f:
        for line in f:
            fields = line.split('|')
            us__ages[int(fields[0])] = int(fields[1])
            
    return us__ages

def loadMovieNames():
    movieNames = {}
    with open("ml-100k/u.ITEM", encoding='ascii', errors='ignore') as f:
        for line in f:
            fields = line.split('|')
            movieNames[int(fields[0])] = fields[1]
            
    return movieNames

def MovieGenre():
    movieGenres = {}
    with open("ml-100k/u.ITEM", encoding='ascii', errors='ignore') as f:
        for line in f:
            fields = line.split('|')
            movieGenres[int(fields[0])] = fields[5]+fields[6]+fields[7]+fields[8]+fields[9]+fields[10]+fields[11]+fields[12]+fields[13]+fields[14]+fields[15]+fields[16]+fields[17]+fields[18]+fields[19]+fields[20]+fields[21]+fields[22]+fields[23]
            
    return movieGenres

#Python 3 doesn't let you pass around unpacked tuples,
#so we explicitly extract the ratings now.
def makePairs( userRatings ):
    
    ratings = userRatings[1]
    (movie1, rating1) = ratings[0]
    (movie2, rating2) = ratings[1]
    return ((movie1, movie2), (rating1, rating2))

def filterDuplicates( userRatings ):
    ratings = userRatings[1]
    (movie1, rating1) = ratings[0]
    (movie2, rating2) = ratings[1]
    return movie1 < movie2



conf = SparkConf().setMaster("local[*]").setAppName("MovieSimilarities")
sc = SparkContext(conf = conf)

print("\nLoading movie names...")
nameDict = loadMovieNames()
genDict = MovieGenre()
data = sc.textFile("file:///SparkCourse/ml-100k/u.data")
us_data = sc.textFile("file:///SparkCourse/ml-100k/u.user")
# Map ratings to key / value pairs: user ID => movie ID, rating
ratings = data.map(lambda l: l.split()).map(lambda l: (int(l[0]), (int(l[1]), float(l[2]))))

# Calculate movie average rating: movieID, avg rating
ratingx = data.map(lambda l: l.split()).map(lambda l: (int(l[1]), float(l[2])))
mv_rt = ratingx.mapValues(lambda x: (x, 1)).reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1]))
mv_avg_rt = mv_rt.mapValues(lambda x:  round(x[0] / x[1],1))

# Calculate user average age by movie: movieID, avg age
# movie ID, user ID
#rdd1 = data.map(lambda l: l.split()).map(lambda l: (int(l[1]),int(l[0])))
# rdd movie ID, user age
#user_ag_dct = Us_age()
#rdd2 = rdd1.mapValues(lambda x: user_ag_dct[x])
# At this point movie ID, age
# Reduce and calculate avg age and generate dict
#rdd3 = rdd2.mapValues(lambda x: (x, 1)).reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1]))
#mv_avg_age = rdd3.mapValues(lambda x:  round(x[0] / x[1],1))
#avg_age_dicts = mv_avg_age.collectAsMap()
# Calculate std and generate dict
#rdd4 = rdd2.map(lambda x: (x[0],((x[1]-avg_age_dicts[x[0]])*(x[1]-avg_age_dicts[x[0]]), 1))).reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1]))
#mv_avg_age2 = rdd4.mapValues(lambda x:  round(np.sqrt(x[0] / x[1]),1))
#avg_age_dicts2 = mv_avg_age2.collectAsMap()
#mvv= mv_avg_age.take(10)
#for i in mvv:
#    print(i)



# Emit every movie rated together by the same user.
# Self-join to find every combination.
joinedRatings = ratings.join(ratings)
# At this point our RDD consists of userID => ((movieID, rating), (movieID, rating))

# Filter out duplicate pairs
uniqueJoinedRatings = joinedRatings.filter(filterDuplicates)

# Now key by (movie1, movie2) pairs.
moviePairs = uniqueJoinedRatings.map(makePairs)

# We now have (movie1, movie2) => (rating1, rating2)
# Now collect all ratings for each movie pair and compute similarity
moviePairRatings = moviePairs.groupByKey()

# We now have (movie1, movie2) = > (rating1, rating2), (rating1, rating2) ...
# Can now compute similarities.
moviePairRatings1 = moviePairRatings.map(lambda x: (x[0],(x[0],x[1])))
moviePairSimilarities = moviePairRatings1.mapValues(compMovGen).cache()

print('one minute')
# Save the results if desired
#moviePairSimilarities.sortByKey()
#moviePairSimilarities.saveAsTextFile("movie-sims")

# Extract similarities for the movie we care about that are "good".

if (len(sys.argv) > 1):

    scoreThreshold = 0.8
    minimumRat = 3
    movieID = int(sys.argv[1])
    genre_similar = 0.5

    # Filter for movies with this sim that are "good" as defined by
    # our quality thresholds above
    filteredResults = moviePairSimilarities.filter(lambda pairSim: \
        (pairSim[0][0] == movieID or pairSim[0][1] == movieID) \
        and pairSim[1][0] > genre_similar and pairSim[1][1] > scoreThreshold)

    # Sort by quality score.
    results = filteredResults.map(lambda pairSim: (pairSim[1], pairSim[0])).sortByKey(ascending = False).take(20)

    print("Top 10 similar movies for " + nameDict[movieID])
    a= 0
    
    for result in results:
        (sim, pair) = result
        # Display the similarity result that isn't the movie we're looking at
        similarMovieID = pair[0]
        if (similarMovieID == movieID):
            similarMovieID = pair[1]
        a1 = genDict[similarMovieID]
        b1 = genDict[movieID]
        movieSc = ratingx.filter(lambda x: x[0]==similarMovieID).collect() 
        tm1 = mv_avg_rt.filter(lambda x: x[0]==similarMovieID).collect()
        v1,v2 = tm1[0]
        tm = int(round(v2,0))
        # Show movie=sci-fi and similar_movie=sci-fi ,movie=drama and similar_movie=drama, movie=action and similar_movie=action
        # ,movie=children and similar_movie=children 
        if tm>=minimumRat:
            s2= 'score: '+str(sim[1])
            s1= 'genre_similarity: '+str(sim[0])
            s3= 'rating '+ "*"*tm
            s4= 'genre similarity '+str(a1)
            sys.stdout.write("%-60s %-25s %-25s %-15s %-15s\n" % (nameDict[similarMovieID],s1,s2,s3,s4))
             
            a+= 1
        if a==10:
            break
        
#13:51- 13:54