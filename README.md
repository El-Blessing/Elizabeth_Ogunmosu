# Elizabeth_Ogunmosu
# 1. Overview
This project is a single-page HTML application with embedded JavaScript that demonstrates numerical integration using the Trapezoidal Rule and Simpson's Rule. The application allows a user to select a predefined mathematical function, define the integration interval 
[a,b], and set a desired accuracy (tolerance).

# 2. Implementation & Architecture
The application is built using Object-Oriented Programming (OOP), specifically utilizing a JavaScript class to encapsulate the integration logic, as encouraged by the assignment guidelines.

The Integrator Class: This class is instantiated with the chosen mathematical function f(x) and the bounds a and b. It contains the core mathematical methods.

User Interface: The HTML/CSS frontend provides a simple form with dropdowns for function selection and number inputs for the interval, initial sub-intervals, and tolerance. It displays the final approximated results cleanly.

# 3. Core Mathematical Functions
Both methods calculate the area under the curve by dividing the interval into nsub-intervals of width 
h = (b−a)/ n

Trapezoidal Rule: Approximates the area by dividing the region into trapezoids.

Formula used: h/2 [f(a) + 2∑f(x_i) + f(b)]

Simpson's Rule: Approximates the area using parabolic arcs, which is generally more accurate for smooth functions. It strictly requires an even number of sub-intervals.

Formula used: h/3 [f(a)+4∑f(x_odd)+2∑f(x_even)+f(b)]


# 4. Automated Step Size Selection
To achieve this accuracy without requiring them to guess the correct number of sub-intervals, the application implements an iterative adaptive algorithm:

The integral is calculated using the initial number of sub-intervals n.

The number of sub-intervals is doubled (n= n × 2), and the integral is calculated again.

The algorithm compares the two results. The absolute difference between them is the estimated error.

If the error is greater than the user-defined tolerance, the loop repeats (doubling n again).

The loop terminates when the difference is smaller than the tolerance, returning the highly accurate result and the total number of intervals used.

# 5. Conclusion
This tool successfully demonstrates that adaptive algorithms can dynamically adjust their resolution to find an accurate approximation of a definite integral. By utilizing Simpson's rule alongside the adaptive step size, the tool can achieve very high precision with fewer computational steps than a static Trapezoidal approach.
