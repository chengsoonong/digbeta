# Revisiting revisits

In recommendation tasks such as trajectory recommendation, it is desirable to avoid revisiting
a state or location that has already been visited before.

## Introduction

- connect to workshop
- distinguish between next location vs whole trajectory
- define word usage: trajectory, path, walk, sequence, tour, etc.
- describe relation to travelling salesman, and say why different
- contributions of this paper

## Christofides' Algorithm

- How to use results from travelling salesman for trajectory recommendation
- Not obvious how to directly use Christofides?
  https://research.googleblog.com/2016/09/the-280-year-old-algorithm-inside.html
  Do we just find double visits and bypass?

## Subtour Elimination using Integer Programming

We should probably ask Phil Kilby or Hassan about the two or three most popular subtour eliminations

- Elementary shortest path problems
  http://www.optimization-online.org/DB_FILE/2014/09/4560.pdf
- Different subtour eliminations
  - (Dantzig, Fulkerson, Johnson, 1954) has 3 different versions
  - (Miller, Tucker, Zemlin, 1960) has another one.
  - Seems to be one more, not sure by whom (maybe Belmore, Malone, 1971)
  http://www.or.unc.edu/~pataki/papers/teachtsp.pdf


## Top-k Best Sequences using List Viterbi

- use list viterbi to go down top-k best sequences until we find one without loops
- two versions of serial list viterbi are equivalent
- parallel list viterbi

## Empirical comparison

- run time
- memory?

## Discussion and Conclusion

- Learning
- diversity, MMR
- See whether any of the workshop topics might have problems that this paper applies.