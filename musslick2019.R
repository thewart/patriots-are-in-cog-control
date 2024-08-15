library()

control_module <- function(t, y, par) {
  with(as.list(par), {
    net1 <- w_self*y[1] + w_lat*y[2] + istask1*in_strength
    net2 <- w_self*y[2] + w_lat*y[1] + (!istask1)*in_strength
    dy1 <- -y[1] + plogis(g*net1)
    dy2 <- -y[2] + plogis(g*net2)
    return(list(c(dy1, dy2)))
  })
}

par <- c(
  w_self=1,
  w_lat=-1,
  in_strength = 0.8,
  istask1=TRUE
)

switch_cost <- function(g, par) {
  steady <- ode(y=c(y1=0.5, y2=0.5), times=seq(0, 100, 0.01), control_module, 
                c(par, g=g)) |> tail(1)
  ode(y=c(y1=steady[,'y1'], y2=steady[,'y2']), times=seq(0, 100, 0.01), 
      control_module, c(par, g=g))
}

