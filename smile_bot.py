import csv
import scipy.io
import numpy as np
from random import shuffle
from math import log
mat = scipy.io.loadmat('labeled_images.mat')
np.seterr(all="raise")
# test = scipy.io.loadmat('public_test_images.mat')

# def view_distribution(d):
#     result = dict()
#     for elem in d:
#         result[d[elem]] = 1 if d[elem] not in result else result[d[elem]] + 1
#     return d

count_identities = lambda l: dict() if not len(l) else add_cur(count_identities(l[1:]),l[0])

def add_cur(d,i):
    if i in d:
        d[i] += 1
        return d
    else:
        d[i] = 1
        return d

class KNN_classifier:
    def __init__(self, images, labels, weights=1, k=3, lambda_=1.0):
        self.big_D = lambda d, w: 1.0/((1+np.exp(-w))*d) if d > 0 else np.inf
        self.ada_D = lambda d, w: w/d if d > 0 else np.inf
        self.k = k
        self.lambda_ = lambda_
        self.lab = labels
        self.wgt = weights
        self.mean = np.mean(images, axis=0)
        self.std = np.std(images, axis=0)
        self.img = self.process_images(images)

    def process_images(self, images):
        # return images
        return (images - self.mean) / self.std

    def classify(self, test_set, test_labels=None, boost="altBoost", weighted_vote=True):
        if len(test_set.shape)==1:
            test_set = np.array([test_set])
        test = self.process_images(test_set)
        cat = np.array([])
        for image in test:
            diff = np.sqrt(((self.img - image)**2).sum(axis=1))*self.lambda_
            sorted_diff = diff.argsort()
            
            ClassCount=dict()
            for i in xrange(self.k):
                vote = self.lab[sorted_diff[i]]
                ClassCount[vote] = ClassCount.get(vote,0) + self.big_D(diff[sorted_diff[i]], self.wgt[sorted_diff[i]]) if boost== "altBoost" else self.ada_D(diff[sorted_diff[i]], self.wgt[sorted_diff[i]])
            cat = np.append(cat, [max(ClassCount, key=lambda x: ClassCount[x])])
        
        if test_labels != None:
            return np.sum(cat == test_labels)/float(test_labels.shape[0])
        return cat 


    def all_against_one(self, boost="altBoost"):
        cat = np.array([])
        indexes = []
        for image in self.img:

            diff = np.sqrt(((self.img - image)**2).sum(axis=1))*self.lambda_
            sorted_diff = diff.argsort()
            
            ClassCount=dict()
            for i in xrange(self.k):
                vote = self.lab[sorted_diff[i+1]]
                ClassCount[vote] = ClassCount.get(vote,0) + self.big_D(diff[sorted_diff[i+1]], self.wgt[sorted_diff[i+1]]) if boost == "altBoost" else self.ada_D(diff[sorted_diff[i+1]], self.wgt[sorted_diff[i+1]])

            cat = np.append(cat, [max(ClassCount, key=lambda x: ClassCount[x])])
            indexes.append(tuple(sorted_diff[1:1+self.k]))
        # error = np.sum(cat == self.lab)/float(self.lab.shape[0])
        return (cat, indexes)

def separate(images, labels, test_size):
    zipped_samples = zip(images.transpose(),labels)

    test = sample(zipped_samples, test_size)
    test_img, test_lab = zip(*test)
    test_img = np.array(test_img).reshape(test_img.shape[0], 1024)
    test_lab = np.array(test_lab).reshape(len(test))
    
    training = filter(lambda x: touple_in_sample(test, x), zipped_samples)
    train_img, train_lab = zip(*training)
    train_img = np.array(train_img).reshape(train_img.shape[0], 1024)
    train_lab = np.array(train_lab).reshape(len(training))

    return (test_img, test_lab, train_img, train_lab)


class adaBoost():
    
    def __init__(self, train_images, train_labels, rounds=30, param=None, learner_weight=None, file_name=None, lambda_=1.0):
        if (not param or not learner_weight) and not file_name:
            self.train_images = train_images
            self.train_labels = train_labels
            self.param = []
            self.learner_weight = []
            weights = np.full(train_labels.shape, 1.0/train_labels.shape[0])
            for i in range(rounds):
                print "round %s" % i
                weights = weights/np.sum(weights)
                self.param.append(weights)
                learner = KNN_classifier(train_images, train_labels, weights,1,lambda_=1.0)
                predictions, k_nearests = learner.all_against_one("adaBoost")
                mask = train_labels == predictions
                error = np.sum(weights[np.invert(mask)])
                print "model error %s" % (error)
                if error > 0.5:
                    break
                beta_r = error/(1-error)
                weights[mask] *= beta_r
                self.learner_weight.append(beta_r)
        elif not file_name:
            self.param = param
            self.learner_weight = learner_weight
            self.train_images = train_images
            self.train_labels = train_labels
        else:
            self.train_images = train_images
            self.train_labels = train_labels
            with open(file_name, "rb") as f:
                reader = csv.reader(f)
                zipped = [(float(line[0]),line[1:]) for line in reader]
                self.learner_weight, param = zip(*zipped)
                self.param = [[] for i in range(len(param))]
                for n, lista in enumerate(param):
                    self.param[n] = [float(i) for i in lista]


    def classify(self, test_images, test_labels=None):
        result = np.array([])
        for image in test_images:
            ClassCount=dict()
            for weights, beta_r in zip(self.param, self.learner_weight):
                vote = KNN_classifier(self.train_images, self.train_labels, weights).classify(image)[0]
                ClassCount[vote] = ClassCount.get(vote,0) + log(1/beta_r)
            result = np.append(result, [max(ClassCount, key=lambda x: ClassCount[x])])
        if test_labels != None:
            return np.sum(result == test_labels)/float(test_labels.shape[0])
        return result

    def out(self, file_name="ada_wgt.csv"):
        with open(file_name, "wb") as f:
            writer = csv.writer(f)
            for lw, p in zip(self.learner_weight, self.param):
                writer.writerow(lw+p)

class altBoost():
    def __init__(self, train_images, train_labels, rounds=30, lambda_=1.0, param=None, file_name=None):
        if not param and not file_name:
            self.train_images = train_images
            self.train_labels = train_labels
            self.param = []
            weights = np.zeros(train_labels.shape)
            for i in range(rounds):
                print "round %s" % i
                self.param.append(weights)
                learner = KNN_classifier(train_images, train_labels, weights)
                predictions, k_nearests = learner.all_against_one("altBoost")
                mask = train_labels != predictions
                if (train_labels == predictions).all():
                    break
                for wrong, value in filter(lambda x: x[1], enumerate(mask)):
                    for near in k_nearests[wrong]:
                        try:
                            if train_labels[near] != train_labels[wrong]:
                                weights[near] -= lambda_/np.sqrt(np.sum((train_images[near]-train_images[wrong])**2))
                            else:
                                weights[near] += lambda_/np.sqrt(np.sum((train_images[near]-train_images[wrong])**2))
                        except FloatingPointError:
                            print elem, wrong
        elif not file_name:
            self.param = param
            self.train_images = train_images
            self.train_labels = train_labels
        else:
            self.train_images = train_images
            self.train_labels = train_labels
            with open(file_name, "rb") as f:
                reader = csv.reader(f)
                self.param = [float(line) for line in reader]

    def classify(self, test_images, test_labels=None):
        result = np.array([])
        for image in test_images:
            ClassCount=dict()
            for weights in self.param:
                vote = KNN_classifier(self.train_images, self.train_labels, weights).classify(image)[0]
                ClassCount[vote] = ClassCount.get(vote,0) + 1
            result = np.append(result, [max(ClassCount, key=lambda x: ClassCount[x])])
        if test_labels != None:
            return np.sum(result == test_labels)/float(test_labels.shape[0])
        return result

    def out(self, file_name="alt_wgt.csv"):
        with open(file_name, "wb") as f:
            writer = csv.writer(f)
            for p in self.param:
                writer.writerow(p)

def partition(total_size, quantity, rev_identity):
    shuffle(rev_identity)
    result = [[] for i in range(quantity)]
    for i in range(quantity-1):
        next_id = rev_identity[-1]
        while len(result[i]) + len(next_id) <= total_size/quantity or (total_size/quantity - len(result[i]))*2 > len(next_id):
            # print len(rev_identity), i
            result[i].extend(next_id)
            rev_identity.pop()
            next_id = rev_identity[-1]
    while len(rev_identity):
        result[quantity-1].extend(rev_identity.pop())
    return result

def find_duplicates(images):
    index = []
    for n, i in enumerate(images):
        flag = False
        for m,j in enumerate(images[n+1:]):
            if (i==j).all():
                index.append((n,m))
                break
    print index

def touple_in_sample(whole_list, touple):
    for elem in whole_list:
        if (elem[0] == touple[0]).all():
            return False
    return True

def process(table, img=False):
    if img:
        table = table.reshape(table.shape[0],table.shape[1]*table.shape[2])
        return np.delete(table, [447,712,713], axis=0)
    table = table.reshape(table.shape[0])
    return np.delete(table, [447,712,713])

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Machine that learns people mood by photos')
    parser.add_argument('method', nargs=1, help="chooses the method to be used", choices=["Ada","Alt"], default="Ada")
    parser.add_argument('-file', nargs=1, help="Analizes data with weights on file", default=[None])
    parser.add_argument('-cross', nargs=1, help="Performs cross validation")
    parser.add_argument('-test', nargs='*', help="Test model with test arguments", default='public_test_images.mat')
    parser.add_argument('-l', nargs='*', type=float, help="Enter with the set of lambdas to be tested", default=[1.0])

    args = parser.parse_args()
    images = process(mat["tr_images"].transpose(), img=True)
    labels = process(mat["tr_labels"])
    identities = process(mat["tr_identity"])

    if args.cross:

        rev_identities = dict()
        for n, ide in enumerate(identities):
            rev_identities[ide] = rev_identities.get(ide, []) + [n]
        rev_identities = rev_identities.values()
        # find_duplicates(images)

        part = partition(labels.shape[0], int(args.cross[0]), rev_identities)
        if args.method[0] == "Ada":
                acc = []
                for i, p in enumerate(part):
                    try:
                        test = part[i]
                        train = reduce(lambda x,y: x+y,map(lambda x: x[1], filter(lambda x: x[0]!= i, enumerate(part))))
                        print "Ada Boost Test %s" % (i+1)
                        adaHip = adaBoost(images[train],labels[train])
                        acc.append(adaHip.classify(images[test],labels[test]))
                        print acc[i]
                        # adaHip.out("ada_r30_%s.csv"%(i))
                    except Exception:
                        print "Couldn't boost"
                acc = np.array(acc)
                print "Ada"
                print "Mean: %s\nStandart Deviation: %s\n" % (np.mean(acc), np.std(acc))
        else:
            for l in args.l:
                acc = []
                for i, p in enumerate(part):
                    test = part[i]
                    train = reduce(lambda x,y: x+y,map(lambda x: x[1], filter(lambda x: x[0]!= i, enumerate(part))))
                    print "Alt Boost Test %s" % (i+1)
                    altHip = altBoost(images[train],labels[train], lambda_=float(l))
                    acc.append(altHip.classify(images[test],labels[test]))
                    print acc[i]
                    # altHip.out("alt_r30_%s.csv"%(i))
                acc = np.array(acc)
                print "Alt with lambda %s" % l
                print "Mean: %s\nStandart Deviation: %s\n" % (np.mean(acc), np.std(acc))
    else:
        test_images = None
        for filename in args.test:
            base = scipy.io.loadmat(filename)
            table = base[filename[:-4]]
            table = table.transpose()
            table = table.reshape(table.shape[0],table.shape[1]*table.shape[2])
            test_images = np.append(test_images, table, axis=0) if test_images != None else table
        # table = "public_test_images"
        # if args.test[0] != 'public_test_images.mat':
        #     table = raw_input("Name the table >\n")
        # test_images = process(accu.transpose(), img=True)
        print test_images.shape
        print "Training "+args.method[0]+" Boost"
        if args.method[0] == "Ada":
            Hip = adaBoost(images, labels, file_name=args.file[0])
        else:
            Hip = altBoost(images, labels, file_name=args.file[0], lambda_=args.l[0], rounds=70)
        if not args.file[0]:
            Hip.out()
        print "Classifying Test with "+args.method[0]+" Boost"
        with open("answer_%s.csv" % (args.method[0]), "wb") as f:
            writer = csv.writer(f)
            writer.writerow(["Id","Prediction"])
            for n, p in enumerate(Hip.classify(test_images)):
                writer.writerow([n+1,int(p)])


    # knn = KNN_classifier(images, labels,np.zeros(labels.shape))
    # print knn.

