(ns stochastic-bandits.core)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;
;; helper functions
;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;; 1.find and indexing functions

(defn indices-of [f coll]
  (keep-indexed #(when (f %2) %1) coll))

(defn rand-index-of [f coll]
  (rand-nth (indices-of f coll)))

(defn find-thing [value coll]
  (rand-index-of #(= % value) coll))

(defn find-max-index [coll] 
  (find-thing (apply max coll) coll))

;; 2.calculate incremental average

(defn inc-average [old-avg new-value n] (+ old-avg (/ (- new-value old-avg) n)))  ;slow

;; 3.parallel computation of mean, central moments and standard deviation

(defn casual-mean [numbers-vec] (/ (apply + numbers-vec) (count numbers-vec))) ;;do not use this for big numbers!

(defn mean [numbers] (let [numbers-vec (vec numbers)] (loop [avg 0 val (numbers-vec 0) i 1] 
                                                        (if (= i   (count numbers-vec)) 
                                                          (inc-average avg val i)
                                                          (recur (inc-average avg val i) 
                                                                 (numbers-vec i) 
                                                                 (inc i))))))

(defn nth-central-moment [in-numbers n] (let [avg (mean in-numbers)] 
                                          (mean (pmap #(Math/pow (- % avg) n) in-numbers))))

(defn variance [in-numbers] (nth-central-moment in-numbers 2))

(defn standard-deviation [in-numbers] (Math/sqrt (variance in-numbers)))

(defn nth-standardized-moment [in-numbers n]
  (/ (nth-central-moment in-numbers n) (Math/pow (standard-deviation in-numbers) n)))

(defn skewness [in-numbers] (nth-standardized-moment in-numbers 3))

(defn kurtosis [in-numbers] (nth-standardized-moment in-numbers 4))

;; 4.normal distribution

(defn box-muller [] (* (Math/sqrt (* -2 (Math/log (rand)))) 
                       (Math/cos (* 2 Math/PI (rand)))))

(defn gaussian-rand [mu sigma] (+ mu (* sigma (box-muller))))  

(defn get-gaussian-func [mu sigma] #(gaussian-rand mu sigma))

;; 5.sample element from probability indexed collection
;;   returns (elems n) with probability of (probs n)
;;   used for the general strategy function

(defn sum-up [v] (map #(apply + (take % v)) (range 1 (inc (count v)))))

(defn sample-indexed-dist [probs elems]
  (let [total-prob (apply + probs)
        guess (rand total-prob)
        intervalls (sum-up probs)] 
    (elems (first (indices-of #(< guess %) intervalls)))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;
;; to be accumulated base-algorithms
;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;


;; 1. epsilon greedy, straight forward

(defn greedy-acc "accumulator function for Epsilon-greedy algorithm"
  [[bandits averages income n epsilon]]
  (let [safe-guess? (> (rand) epsilon)
        guessed-bandit (rand-int (count bandits))
        ;;guessed-value ((bandits guessed-bandit))
        maxed-bandit-index (find-max-index averages)
        ;;maxed-bandit-outcome ((bandits maxed-bandit-income))
        choosen-index (if safe-guess? maxed-bandit-index guessed-bandit)
        outcome ((bandits choosen-index))]
    [bandits
    ;;(assoc averages choosen-index (inc-average (averages choosen-index) outcome n))
     (update-in averages [choosen-index] inc-average outcome n)
     (+ income outcome)
     (inc n)
     epsilon]))

;; 2. accumulation algorithm for probability tables and strategy
;;    - more general with a functional style, 
;;    - lets implement other algorithms easier

(defn probbandit-chooser [bandits probs] 
  (let [chosen-bandit-ind (sample-indexed-dist probs (vec (range (count bandits))))
        outcome ((bandits chosen-bandit-ind))] 
    {:index chosen-bandit-ind,
     :outcome outcome}))

(defn general-acc
  [[bandits strategy averages old-probs played-arms income n]]
  (let [probs (strategy averages old-probs played-arms)
        chosen-bandit (probbandit-chooser bandits probs)] 
    [bandits
     strategy
     (update-in averages [(chosen-bandit :index)] inc-average (chosen-bandit :outcome) n)
     probs
    ;; (assoc averages (choosen-bandit :index) 
    ;;        (inc-averages (averages (choosen-bandit :index)) (choosen-bandit :outcome) n))

     ;;(assoc played-arms (chosen-bandit :index) (inc (played-arms (chosen-bandit :index))))
     (update-in played-arms [(chosen-bandit :index)] inc)
     (+ income (chosen-bandit :outcome))
     (inc n)]))

;; 3.some strategies for the general algorithm, they all take some 
;;   input parameters and return a probability table for the bandits
;;   the getters return a strategy function 
;;   -not every algo uses every argument

;; epsilon-greedy
(defn greedy-strategy [averages old-probs played-arms epsilon]
  (let [max-prob-ind (find-max-index averages)
        bandit-cnt (count averages)]
    (mapv #(if (= % max-prob-ind) (+ 1 (- epsilon) (/ epsilon bandit-cnt)) 
               (/ epsilon bandit-cnt)) 
          (range (count averages)))))

(defn get-greedy-strategy [epsilon] #(greedy-strategy %1 %2 %3 epsilon))

;; boltzmann is also known as softmax
(defn boltzmann-strategy [averages old-probs played-arms tau]
  (let [exp-part #(Math/exp (/ % tau))]
    (mapv #(/ (exp-part %) (apply + (map exp-part averages))) averages)))

(defn get-boltzmann-strategy [tau] #(greedy-strategy %1 %2 %3 tau))

;; standard pursuit algorithm Sutton Barto
(defn pursuit-strategy [averages old-probs played-arms beta]
  {:pre [(< 0 beta) (< beta 1)]}
  (let [max-prob-ind (find-max-index averages)
        bandit-cnt (count averages)]
    (mapv #(if (= % max-prob-ind) 
             (+ (old-probs %) (* beta (- 1 (old-probs %)))) 
             (* beta (- 0 (old-probs %)))) 
          (range (count averages)))))

(defn get-pursuit-strategy [beta] #(pursuit-strategy %1 %2 %3 beta))

;; todo: reinforcement and UCB

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;
;; final algorithms for fast calculations without storing intermediate values
;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;; 1. epsilon greedy, straight forward

(defn greedy-rec "recursive function for Epsilon-greedy algorithm"
  [[bandits average income n epsilon]]
  (loop [in [bandits average income 1 epsilon]] 
    (if (= (in 3) n) in (recur (greedy-acc in)))))

;; 2. general algorithm for different strategies

(defn general-rec
  [[bandits strategy averages  old-probs played-arms income n]]
  (loop [in [bandits strategy averages  old-probs played-arms income 1]]
    (if (= (in 6) n) in (recur (general-acc in)))))



