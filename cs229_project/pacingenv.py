import numpy as np

        
class PacingEnv():
        
    def __init__(self, dt = 5, dailyBudget = 100, ctrThres = 0, maxBid = 20, cpcGoal = 100, now = 0, index = 0, maxctr = 0.02, basecost = 10.0, baseIntensity = 5.0):
        self.numslots= 24*60 # for 1 day
        self.dt = dt #simu time interval dt minutes
        self.maxctr = maxctr
        self.basecost = basecost
        self.baseIntensity = baseIntensity
        self.pacingcache = []
        self.cache = {}
        self.dailyBudget = dailyBudget
        #biding control parameters
        self.ctrThres = ctrThres
        self.maxBid = maxBid
        self.cpcGoal = cpcGoal
        #time step in minutes, its max val i self.numslots
        self.now = now
        #generate random data
        self.generateData()
        #index of data points
        self.index = index
        #metric stats
        self.impCnt = 0
        self.clickCnt = 0
        #np array specifying range of values in [low,high] format
        self.action_space = np.array([[0.0, 1.0]])
        self.observation_space = np.array([[0.0,dailyBudget],[0.0,self.numslots/self.dt]])
        self.state = np.array([dailyBudget,0],dtype = float)
        curve = np.zeros(288)
        for i in range(288):
            curve[i] = i/288.0 + np.sin(2*np.pi*i/288.0)
        self.spendingCurve = curve
        #self.seed()
        
    def generateData(self):
        """
        generate impressions of given size
        
        numslots: number of time slots
        timestamp: starting from 1543000, in minutes
        click: 1 or 0
        baseIntesenty: how many imps per minute, on average
        ctr: [0,maxctr) float
        Ts: the period assuming temporal granularity for impression is minute
        phi: temporal shift to create reasonable TOD pattern
        cost: in [basecost,basecost*5) range
        """
        numslots = self.numslots
        maxctr = self.maxctr
        basecost = self.basecost
        baseIntensity = self.baseIntensity
        #TOD pattern
        tstep = np.array([i for i in range(numslots)])
        Ts = 1/(24*60.0)
        shape = (np.sin(2*3.14159*Ts*(tstep-10*60))+1.5)*0.8
        #num of imps at each time point
        lengths=[]
        for s in shape:
            lengths.append(int(s*baseIntensity))
        #total num of imps
        length = sum(lengths)
        
        #index to timeslot map
        lookup = np.cumsum(lengths)
        #we focus on pacing not bidding algorithm so clicks can be generated without correlation to ctr
        click = np.random.randint(2, size=length)
        ctr = np.random.random_sample(length)*maxctr
        cost = np.multiply((np.random.random_sample(length)*basecost*4+basecost),np.sqrt(ctr))
        #self.data = np.stack((tstep,click,ctr,cost),axis=-1)
        
        self.data = []
        for i in range(length):
            timeindex = np.argmax(lookup>i)
            line = (tstep[timeindex],click[i],ctr[i],cost[i])
            self.data.append(line)
            
        return

    """
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    """
    
    
    def getReward(self,pacing,spendRatio,timeofday):
        #no penalty range is +/- thres
        """thres = 0.00001
        terminal_factor = 1000.0
        #TODO/ add cost for signal discontinuity
        smooth = 5-abs(self.prevpacing - pacing)#abs(self.prevpacing - pacing)
        #abs diff cost for 1st version
        diff = spendRatio - self.spendingCurve[timeofday]
        
        if abs(diff)>thres:
            reward = abs(diff)-thres+smooth
        else:
            reward = smooth
            
        if timeofday==287:
            reward = terminal_factor*abs(spendRatio-1.0)
        return -reward"""
        smooth = 0
        #smooth = -abs(self.prevpacing - pacing)
        """
        #second order smoothness condition
        diff1 = self.prevpacing2 - self.prevpacing
        diff = self.prevpacing - self.pacing
        smooth = abs(diff1-diff)
        """
        #smooth = abs(self.prevpacing - self.pacing)/5.0
        #time sensitive cost
        factor = 100*(timeofday/288.0)**2
        if timeofday>=280:
            factor = factor*10
        timeofday = int(timeofday)
        return -factor*(spendRatio - self.spendingCurve[timeofday])**2-smooth
    
    
    def step(self,pacing):
        #only simulate for one day
        if self.now >= 288*self.dt:
            return None, None, True, {}
            
        #fetch the impressions within step
        imps = []
        #imp[0]~imp[3], timestamp, click, pctr, cost
        while self.index<len(self.data):
            curImp = self.data[self.index]
            imps.append(curImp)
            if curImp[0] >= self.now + self.dt:
                break
            self.index = self.index+1
        ###auction, then update states and compute rewards
        impCnt = 0
        cost = 0.0
        clickCnt = 0
        for imp in imps:
            rnd = np.random.uniform()
            if rnd>pacing or imp[2]<self.ctrThres:#campaign won't join auction
                continue
            #auction
            if self.auction(imp,self.cpcGoal,self.maxBid): #won auction
                impCnt = impCnt + 1
                cost = cost + imp[3]
                clickCnt = clickCnt + imp[1]
        
        #update states
        self.state[0] = self.state[0]-cost#budget
        self.impCnt = self.impCnt + impCnt#imps
        self.clickCnt = self.clickCnt + clickCnt#clickCnt
        
        self.cache[self.now] = (self.cpcGoal,self.ctrThres,self.maxBid,pacing,cost,impCnt,clickCnt)
        
        #update time
        self.now = self.now + self.dt
        
        #assemble the state and reward then return
        time_left = 288-(self.now/self.dt)
        self.state[1] = 288-time_left-1
        #remainBudgetRatio = max(0,self.state[0]/float(self.dailyBudget))#remaining daily budget
        #state = self.discretizeState(0,0,time_left,remainBudgetRatio)
        #state = self.discretizeState(time_left,remainBudgetRatio)
        self.pacing = pacing
        self.pacingcache.append(self.pacing)
        reward = self.getReward(pacing,self.state[0]/float(self.dailyBudget),self.state[1])
        isDone = False
        if time_left<=0:
            isDone = True
        return self.state, reward, isDone, {}
    
    def auction(self,imp,cpcGoal,maxBid):
        bid = min(imp[2]*cpcGoal,maxBid) 
        if bid>=imp[3]:
            return True
        else:
            return False

    def reset(self):
        #regenerate data
        self.generateData()
        self.state = np.array([self.dailyBudget,0])
        self.now = 0L
        self.index = 0
        self.impCnt = 0
        self.clickCnt = 0
        self.dataSize = len(self.data)
        #time_left = 288-(self.now/self.dt)
        #remainBudgetRatio = max(0,self.state[0]/float(self.dailyBudget))
        self.pacing = None
        self.pacingcache = []
        #state = self.discretizeState(0.0,0.0,time_left,remainBudget)
        return self.state
