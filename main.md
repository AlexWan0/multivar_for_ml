# Enough Multivariable Calc for Basic Deep Learning
Assumes basic knowledge about vectors

With Calc I and II, you learned how to take integral and derivatives of single variable functions.

That is functions that took a single input in, and produced a single value out.

In ML, however, you largely deal with multivariable functions. These functions may take multiple inputs and may have multiple outputs.

For example, an image classifier that classifies "hot dog" or "not hot dog" takes in image input (a vector that may have 128 * 128 * 3) values, and produces a single values from 0 to 1.

Let's say, if x_vec is a hot dog, it would output y = 1 and if it was not a hot dog, it would output y = 0.

f(x_vec) = y where x_vec is a vector of size 128 * 128 * 3.
You can imagine this as f(x_1, x_2, x_3, ..., x_49152) = y

This model may also have parameters that change (e.g. the parameters for a linear regression model might be its slope and y-intercept), in that case there would be even more "inputs" to the function, as changing those parameters would change the model ouptut.

f(x_vec, w_1, w_2, ...) = y

# Partial Derivatives
A common problem you'll encounter in ML is "how does changing the parameters change the output". Intuitively, if you want your model to perform accurately, then you would want to change the parameters such that the output is correct.

Well if the function only had one input and one output, then that would would be easy -- it's just the derivative.

Let's pretend our image classifier is f(w_1) = y. The derivative dy/dw_1 us that for a (infinitesimally) small increase in w_1, there should be a dy/dw_1 rate of change in y (approximately, in the area surrounding w_1 - recall: the definition of differentiability, which tells us that if a function is differentiable at some point then, at the infinitesimally small area surrounding that point, the slope is the same).

How does this apply to our actual function, which has many inputs? Well, it tells us the same thing -- that increasing w_1 would increase y at rate dy/dw_1, we're just pretending all the other parameters are constant! This is called the *partial derivative* of y with respect to w_1.

Let's look at a simple example where y = w_1 * w_2. In this case our function is f(w_1, w_2) = y
The partial derivative of y wrt w_1 is w_2, and the partial derivative of y wrt w_2 is w_1.

In other words, if w_2 is constant, then increasing w_1 by a small amount would increase y at a rate of w_2, and vice versa. If this doesn't make sense, pretend w_2 is some random number (like 2), and imagine taking a regular derivative.

# Gradients
Now that we have our partial derivatives for each input to the function, how do we describe our multivariable function as whole? One way is to put them into a vector.

For our simple example above, f(w_1, w_2), we could write:
<partial of y wrt w_1, partial of y wrt w_2>
= <w_2, w_1>

This is called the *gradient*.

# Directional Derivatives
Let's think look at the machine learning problem from before:
"How does changing the parameters change the output?"

Let's first make this a bit more specific.

"How does changing the parameters change the output *for some given input*?" That is, let's try to optimize our model one image at a time.

In that case, we don't really care about how our input changes anymore, and so our function looks like this:

f(w_1, w_2, ...) = y

And our gradient is this:

grad_y = <partial of y wrt w_1, partial of y wrt w_2, ...>

Let's say that we know our given input is a hot dog, so in that case, we want our output y to increase (towards y = 1) -- how should we change our parameters w to do this?

For this, we need to think about *directional derivatives*: denoted as D_u f(w_vec).

It tells us, if were to change w_vec in the direction of some unit vector u, how would the scalar f(w_vec) change.

But hey! This sounds familiar -- remember in single dimension Calc what a derivative was for some df(x)/dx: if we were to change x, how would f(x) change. A directional derivative is a generalization of a regular derivative, except we can move x in any arbitrary direction instead of just larger/smaller!

Let's decompose this using what we know from single dimension Calc: we know from before how changing each individual w_1 affects f(w_vec) (the partial derivative). We know that the direction we're interested in is u, so if were interested in w_1's effects on f(w_vec) in the direction of u, we would just take a step in the direction of u_1.

More generally, assuming the other dimensions are constant, increasing or decreasing w_i changes f(w_vec) by:
partial of f(w_vec) wrt w_i * u_i.

So if we wanted to know how *all* w_vec changes f(w_vec) in the direction of u -- that is, D_u f(w_vec), we can just sum them!

But why are we allowed to do this? Remember that our partial derivatives were under the assumption that the other inputs were *constant* -- if we take a small step along w_1 to change f(w_vec) at the rate partial f(w_vec) wrt w_1, how do we know that f(w_vec) would still change at the rate partial f(w_vec) wrt w_2 now that we've changed w_1?

Recall: the definition of differentiability from single variable Calc: If f(x) is differentiable at x', there must be a constant slope in the (infinitesimally) small area surrounding x'. This constant slope is the derivative -- the constant rate of change when you increase/decrease x by an (infinitesimally) small amount. This generalizes to multivariable functions too, except this "small area", instead of just being the positive/negative direction, covers every direction. And this slope is the gradient -- it tells us the constant slope along each dimension, instead of just one. That is, for some arbitary index i of gradient grad f(w_vec), it gives us the constant rate of change when you increase/decrease w_i by an (infinitesimally) small amount.

This means that, as long as we're taking infinitesimally small steps, our partial derivatives would, in fact, stay constant -- which means that we can just sum them!

So, our directional derivative D_u f(w_vec) = partial wrt w_1 * u_1 + partial wrt w_1 * u_2 ... = grad w_vec dot u

This is important because we know from basic lin alg that the magnitude of the dot product of two vectors = |grad w_vec||u|cos theta where theta is the angle between the two vectors. By assumption u is a unit vector (because we only care about its direction) and grad w_vec is constant (with respect to the direction u that we're choosing), so to maximize this value, we want cos theta to be 1, meaning that theta is 0. This is really nice! Because if we want to find the direction (u) such that we're increasing f(w_vec) the fastest (maximum of D_u f(w_vec)), we just take a step in the direction of the gradient! We can make the same argument for minimization as well, where the direction where we're decreasing f(w_vec) the fastest is the direction *opposite* of the gradient.

# Vector valued functions
In single valued Calc, we saw functions with a single input and a single output.
We just looked at functions with multiple inputs and a single output.
Of course, we can also have functions that have multiple (or a single) input and multiple outputs.

If we have two inputs and two outputs we can visualize them as vector fields:

# Chain rule
Chain rule extends very naturally to multivariable functions.

Let's just take a look at a few examples:

Let's say we have f(x, y) that outputs a scalar.
We know from before how f changes wrt a small change from x and y:
partial wrt x, partial wrt y

How about f(g(x), g(y))? Where g output a scalar value
For the partial derivative of f wrt x:
First find partial of f wrt g(x), then multiply by g(x) wrt x

How about the partial of f(h(z)) wrt z? Where h(z) outputs a vector.
(e.g. h(z) = <z, z>, and f(x, y) = xy)

Let's split this up by dimensions:
f(h(z)) just equals to f(h(z)_0, h(z)_1)

There's two inputs to f -- we'll just worry about the first one h(z)_0 for now.
We can find the partial of f wrt h(z)_0 (this is just the partial derivative example from before)
We can find the derivative of h(z)_0 wrt z (this is just a single valued function!)

By the chain rule, we can multiply them together:
df/d(h(z)_0) * d(h(z)_0)/dz

This is: how f changes to its first input * how h changes relative to its only input.

But we still need to incorporate f's second input. Following the same procedure, we can find that it equals:
df/d(h(z)_1) * d(h(z)_1)/dz

So how do we combine them? Remember from before that we can just add them!
So the total rate at which f changes relative to z is:
df/d(h(z)_0) * d(h(z)_0)/dz + df/d(h(z)_1) * d(h(z)_1)/dz

Intuitively, we can think of this as summing two "paths" to figure out f changes relative to z. z can affect f along its (f's) first input or its (f's) second input, so we find those individually then sum them.

