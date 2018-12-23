# CVaR

Intro:
This program tends to calculate CVaR which is an enhancement  to VaR. The idea
is to calculate the risk that a portfolio is exposed to.
 
It uses the past observed distribution of portfolio returns to estimate what 
your future losses might be at difference likelihood levels and picks the
best combination of tickers to make you richer than others!

The likelihood level chosen is 99%. There is a chance of 1% error given the 
assumptions.

This back-tester starts by training on one month data provided a correct start
date is given. (see point 3 in important notes)
After each iteration the back tester gets another extra month of data to train
itself and pick the best combination.

Important Notes:
1) reset allocation if you re-run the program, as the allocation changes
   while back-testing and you might end up working out incorrect results.
   
2) The script might contain errors and it does. They will be sorted out
    in the newer version.
    
3) start date should always be a first date of month eg. 01/01/2017
   if given 30/12/16 back tester would only have 1-day data to analyze
   and select a best combination of stock in the first iteration.
   
4) number of stocks selected is a hyper parameter which needs to be tuned as per
   your risk appetite
    
I AM NOT LIABLE FOR ANY LOSSES INCURRED!