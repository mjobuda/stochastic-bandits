(defproject org.clojars.waschbar/stochastic-bandits "0.1.0-SNAPSHOT"
  :description "library with stochastic bandit algorithms"
  :url "https://github.com/mjobuda/stochastic-bandits"
  :license {:name "EPL-2.0 OR GPL-2.0-or-later WITH Classpath-exception-2.0"
            :url "https://www.eclipse.org/legal/epl-2.0/"}
  :dependencies [[org.clojure/clojure "1.10.0"]]
  :repl-options {:init-ns stochastic-bandits.core}
   :deploy-repositories [["releases"  {:sign-releases false :url "https://clojars.org"}]
                        ["snapshots" {:sign-releases false :url "https://clojars.org"}]])
