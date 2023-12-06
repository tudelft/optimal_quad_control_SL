#####################################################################################
#
# Problem:  Optimal Control Problem (OCP)
# Dynamics: Equations of motion from (bebop_12dof_thrust.pdf /.tex)
# Transcription: Hermite-Simpson
#
# Author: Dario Izzo (Nov 2018) Modified by Robin Ferede (Sep 2020)
#
#####################################################################################

#Sets---------------------------------------------------------
    set vI := 1..3;
    set vJ := 1..4;
#-------------------------------------------------------------

#Parameters---------------------------------------------------
#Generic
    param n default          100;                   # Number of nodes
    param g default          9.81;                  # [m/s^2] Gravitational Acceleration
    param epsilon default    0.01;                  # Tunes the aggressivity of the optimal solution
    param pi :=              4*atan(1);             # pi
    param K  :=              pi/30;                 # from rpm to rad/s

#Quadrotor params
    param mass default         0.389;               # [kg] Mass
    
    param Ixx default          0.00090600;          # [kg m^2] Moment of Inertia Tensor Component - xx
    param Iyy default          0.00124200;          # [kg m^2] Moment of Inertia Tensor Component - yy
    param Izz default          0.00205400;          # [kg m^2] Moment of Inertia Tensor Component - zz

    param k_p      default  1.41193310e-09; #1.83796309e-09;
    param k_pv     default  -7.97101848e-03;
    param k_q      default  1.21601884e-09; #1.30258126e-09;
    param k_qv     default  1.29263739e-02;
    
    param k_r1     default 2.57035545e-06; #2.24736889e-10;
    param k_r2     default 4.10923364e-07; #3.51480672e-07;
    param k_rr     default 8.12932607e-04; #1.95277752e-03;
    
    param k_x      default 1.07933887e-05; #9.94601020e-06;
    param k_y      default 9.65250793e-06; #1.12924575e-05;
    
    param k_omega  default 4.36301076e-08; #4.33399301e-08;
    param k_z      default 2.78628990e-05; #2.25247782e-05;
    param k_h      default 6.25501332e-02; #5.12153499e-02;
    
    param Mx_ext default  0;
    param My_ext default  0;
    param Mz_ext default  0;
    
    param maxthrust default    1.4*g;                # [N] Max thrust acceleration
    param minthrust default    0.6*g;                # [N] Min thrust acceleration
    
    param omega_max default    12000; #sqrt(maxthrust/(k_omega*4)); #9800;   # [rpm] Max min per rotor
    param omega_min default    3000; #sqrt(minthrust/(k_omega*4)); #3000;   # [rpm] Min rpm per rotor
    
    param omega_hover default  sqrt(g/(k_omega*4));
    param u_hover default      (omega_hover-omega_min)/(omega_max-omega_min);
    
    param tau default          0.06;                 # [s] First order delay constant for controls
    
#State constraint params
    param maxphi   default 80*pi/180;
    param maxtheta default 80*pi/180;
    
#Initial conditions
    param dx0 default          0.0;        # [m] Initial dx
    param dy0 default          0.0;        # [m] Initial dy
    param dz0 default          0.0;        # [m] Initial dz
    
    param vx0 default          0.0;        # [m/s] Initial vx
    param vy0 default          0.0;        # [m/s] Initial vy
    param vz0 default          0.0;        # [m/s] Initial vz
    
    param phi0 default         0.0;        # [rad] Initial phi
    param theta0 default       0.0;        # [rad] Initial theta
    param psi0 default         0.0;        # [rad] Initial psi
    
    param p0 default           0.0;        # [rad] Initial phi rate
    param q0 default           0.0;        # [rad] Initial theta rate
    param r0 default           0.0;        # [rad] Initial psi rate
    
    param utau0{vJ} default    0.0;        # Initial rotor rpm
    
#Final conditions
    param dxn default          0.0;        # [m] Final dx
    param dyn default          0.0;        # [m] Final dy
    param dzn default          0.0;        # [m] Final dz
    
    param vxn default          0.0;        # [m/s] Final vx
    param vyn default          0.0;        # [m/s] Final vy
    param vzn default          0.0;        # [m/s] Final vz
    
    param phin default         0.0;        # [rad] Final phi
    param thetan default       0.0;        # [rad] Final theta
    param psin default         0.0;        # [rad] Final psi
    
    param pn default           0.0;        # [rad] Final phi rate
    param qn default           0.0;        # [rad] Final theta rate
    param rn default           0.0;        # [rad] Final psi rate
    
#Other
    param tn default 2.0;                  # [s] Guess for the final time
#-------------------------------------------------------------

#Sets---------------------------------------------------------
    set I := 1..n;
    set J := 1..n-1;
#-------------------------------------------------------------

#Variables---------------------------------------------------
    var dx {i in I};
    var dy {i in I};
    var dz {i in I};
    
    var vx {i in I};
    var vy {i in I};
    var vz {i in I};
    
    var phi {i in I},   >= -maxphi,   <= maxphi;
    var theta {i in I}, >= -maxtheta, <= maxtheta;
    var psi {i in I};
    
    var p {i in I};
    var q {i in I};
    var r {i in I};
    
    var utau {i in I, j in vJ};
    
    var u  {i in I, j in vJ}, >=0, <=1; 
    #var um {i in J, j in vJ}, >=0, <=1;
    
    var v_final, >= 5;
    #var max_distance, <= 0.3^2;

#-------------------------------------------------------------

#Time variables-----------------------------------------------
    var tf, >=0;
    var dt = tf/(n-1);
    var timegrid{i in I} = dt*(i-1);
#-------------------------------------------------------------



#Dynamic at the grid points-----------------------------------

    # rpm
    var omega{i in I, j in vJ} = omega_min + utau[i, j]*(omega_max-omega_min);
    
    var f1{i in I} = -q[i]*dz[i] + r[i]*dy[i] - vx[i];
    var f2{i in I} =  p[i]*dz[i] - r[i]*dx[i] - vy[i];
    var f3{i in I} = -p[i]*dy[i] + q[i]*dx[i] - vz[i];
    
    var f4{i in I} = -q[i]*vz[i] + r[i]*vy[i] - g*sin(theta[i]) - k_x*vx[i]*sum{j in vJ} omega[i, j];
    var f5{i in I} =  p[i]*vz[i] - r[i]*vx[i] + g*cos(theta[i])*sin(phi[i]) - k_y*vy[i]*sum{j in vJ} omega[i, j];
    var f6{i in I} = -p[i]*vy[i] + q[i]*vx[i] + g*cos(theta[i])*cos(phi[i]) - k_z*vz[i]*sum{j in vJ} omega[i, j] - k_omega*sum{j in vJ} omega[i, j]^2 - k_h*(vx[i]^2 + vy[i]^2);

    var f7{i in I} = p[i] + q[i]*sin(phi[i])*tan(theta[i]) + r[i]*cos(phi[i])*tan(theta[i]);
    var f8{i in I} = q[i]*cos(phi[i]) - r[i]*sin(phi[i]);
    var f9{i in I} = q[i]*sin(phi[i])/cos(theta[i]) + r[i]*cos(phi[i])/cos(theta[i]);
    
    var f13{i in I, j in vJ} = (u[i,j] - utau[i, j])/tau;
    
    # moments
    var taux{i in I} = k_p*(omega[i,1]^2 - omega[i,2]^2 - omega[i,3]^2 + omega[i,4]^2) + k_pv*vy[i] + Mx_ext;
    var tauy{i in I} = k_q*(omega[i,1]^2 + omega[i,2]^2 - omega[i,3]^2 - omega[i,4]^2) + k_qv*vx[i] + My_ext;
    var tauz{i in I} = k_r1*(-omega[i,1] + omega[i,2] - omega[i,3] + omega[i,4]) + 
                       k_r2*(omega_max-omega_min)*(-f13[i,1] + f13[i,2] - f13[i,3] + f13[i,4]) - k_rr*r[i] + Mz_ext;
    
    var f10{i in I} = (q[i]*r[i]*(Iyy-Izz) + taux[i])/Ixx;
    var f11{i in I} = (p[i]*r[i]*(Izz-Ixx) + tauy[i])/Iyy;
    var f12{i in I} = (p[i]*q[i]*(Ixx-Iyy) + tauz[i])/Izz;
    
#-----------------------------------------------------------------------

#State definition at mid-points via Hermite interpolation---------------
    var dxm{i in J}     =   (dx[i] +     dx[i+1])/2 + tf/(n-1)/8 * (f1[i] - f1[i+1]);
    var dym{i in J}     =   (dy[i] +     dy[i+1])/2 + tf/(n-1)/8 * (f2[i] - f2[i+1]);
    var dzm{i in J}     =   (dz[i] +     dz[i+1])/2 + tf/(n-1)/8 * (f3[i] - f3[i+1]);
    
    var vxm{i in J}     =   (vx[i] +    vx[i+1])/2 + tf/(n-1)/8 * (f4[i] - f4[i+1]);
    var vym{i in J}     =   (vy[i] +    vy[i+1])/2 + tf/(n-1)/8 * (f5[i] - f5[i+1]);
    var vzm{i in J}     =   (vz[i] +    vz[i+1])/2 + tf/(n-1)/8 * (f6[i] - f6[i+1]);
    
    var phim{i in J}    =   (phi[i]   + phi[i+1])/2   + tf/(n-1)/8 * (f7[i] - f7[i+1]);
    var thetam{i in J}  =   (theta[i] + theta[i+1])/2 + tf/(n-1)/8 * (f8[i] - f8[i+1]);
    var psim{i in J}    =   (psi[i]   + psi[i+1])/2   + tf/(n-1)/8 * (f9[i] - f9[i+1]);
    
    var pm{i in J}      =   (p[i] + p[i+1])/2   + tf/(n-1)/8 * (f10[i] - f10[i+1]);
    var qm{i in J}      =   (q[i] + q[i+1])/2   + tf/(n-1)/8 * (f11[i] - f11[i+1]);
    var rm{i in J}      =   (r[i] + r[i+1])/2   + tf/(n-1)/8 * (f12[i] - f12[i+1]);
    
    var utaum{i in J, j in vJ} = (utau[i, j] + utau[i+1, j])/2 + tf/(n-1)/8 * (f13[i, j] - f13[i+1, j]);
    var um{i in J, j in vJ} = (u[i, j] + u[i+1, j])/2;
#-----------------------------------------------------------------------

#Dynamic at the mid-points----------------------------------------------

    # rpm
    var omegam{i in J, j in vJ} = omega_min + utaum[i, j]*(omega_max-omega_min);
    var omega_sm{i in J} = sum{j in vJ} omegam[i, j]^2;   
    
    var f1m{i in J} = -qm[i]*dzm[i] + rm[i]*dym[i] - vxm[i];
    var f2m{i in J} =  pm[i]*dzm[i] - rm[i]*dxm[i] - vym[i];
    var f3m{i in J} = -pm[i]*dym[i] + qm[i]*dxm[i] - vzm[i];
    
    var f4m{i in J} = -qm[i]*vzm[i] + rm[i]*vym[i] - g*sin(thetam[i]) - k_x*vxm[i]*sum{j in vJ} omega[i, j];
    var f5m{i in J} =  pm[i]*vzm[i] - rm[i]*vxm[i] + g*cos(thetam[i])*sin(phim[i]) - k_y*vym[i]*sum{j in vJ} omega[i, j];
    var f6m{i in J} = -pm[i]*vym[i] + qm[i]*vxm[i] + g*cos(thetam[i])*cos(phim[i]) - k_z*vzm[i]*sum{j in vJ} omega[i, j] - k_omega*sum{j in vJ} omega[i, j]^2 - k_h*(vxm[i]^2 + vym[i]^2);

    var f7m{i in J} = pm[i] + qm[i]*sin(phim[i])*tan(thetam[i]) + rm[i]*cos(phim[i])*tan(thetam[i]);
    var f8m{i in J} = qm[i]*cos(phim[i]) - rm[i]*sin(phim[i]);
    var f9m{i in J} = qm[i]*sin(phim[i])/cos(thetam[i]) + rm[i]*cos(phim[i])/cos(thetam[i]);
    
    var f13m{i in J, j in vJ} = (um[i,j] - utaum[i, j])/tau;
    
    # moments
    var tauxm{i in J} = k_p*(omegam[i,1]^2 - omegam[i,2]^2 - omegam[i,3]^2 + omegam[i,4]^2) + k_pv*vym[i] + Mx_ext;
    var tauym{i in J} = k_q*(omegam[i,1]^2 + omegam[i,2]^2 - omegam[i,3]^2 - omegam[i,4]^2) + k_qv*vxm[i] + My_ext;
    var tauzm{i in J} = k_r1*(-omegam[i,1] + omegam[i,2] - omegam[i,3] + omegam[i,4]) + 
                        k_r2*(omega_max-omega_min)*(-f13m[i,1] + f13m[i,2] - f13m[i,3] + f13m[i,4]) - k_rr*rm[i] + Mz_ext;
    
    var f10m{i in J} = (qm[i]*rm[i]*(Iyy-Izz) + tauxm[i])/Ixx;
    var f11m{i in J} = (pm[i]*rm[i]*(Izz-Ixx) + tauym[i])/Iyy;
    var f12m{i in J} = (pm[i]*qm[i]*(Ixx-Iyy) + tauzm[i])/Izz;
    
#-----------------------------------------------------------------------

#Objective----------------------------------------------------

        # For power, minimize Simpson's approximation to the integral:
        #
        #        \int{ f(t)dt }
        #     ~= \sum_{  dt/6 * f(t) + 4*f(t+dt/2)  + f(t+dt)  }
        #               for t=(dt,2*dt,3*dt...)
        #cost has the values at t = i*dt
        #costm has the values at t = i*dt + dt/2
        
    var cost{i in I}  = sum{j in vJ} (u[i,j])^2;
    var costm{i in J} = sum{j in vJ} (um[i,j])^2;
    
    var smoothing_term = dt/6 * sum{i in J} (cost[i]+4*costm[i]+cost[i+1]);

    minimize myobjective: smoothing_term * epsilon + tf * (1 - epsilon);

#-------------------------------------------------------------

#Simpson Formula---------------------------------------------------------
subject to
    dynamicdx{i in J}:  dx[i]  =    dx[i+1] - tf/(n-1)/6*(f1[i]  + f1[i+1]  + 4*f1m[i]);
    dynamicdy{i in J}:  dy[i]  =    dy[i+1] - tf/(n-1)/6*(f2[i]  + f2[i+1]  + 4*f2m[i]);
    dynamicdz{i in J}:  dz[i]  =    dz[i+1] - tf/(n-1)/6*(f3[i]  + f3[i+1]  + 4*f3m[i]);
    
    dynamicvx{i in J}:  vx[i]  =    vx[i+1] - tf/(n-1)/6*(f4[i]  + f4[i+1]  + 4*f4m[i]);
    dynamicvy{i in J}:  vy[i]  =    vy[i+1] - tf/(n-1)/6*(f5[i]  + f5[i+1]  + 4*f5m[i]);
    dynamicvz{i in J}:  vz[i]  =    vz[i+1] - tf/(n-1)/6*(f6[i]  + f6[i+1]  + 4*f6m[i]);
    
    dynamicp{i in J}: phi[i]   = phi[i+1]   - tf/(n-1)/6*(f7[i] + f7[i+1] + 4*f7m[i]);
    dynamicq{i in J}: theta[i] = theta[i+1] - tf/(n-1)/6*(f8[i] + f8[i+1] + 4*f8m[i]);
    dynamicr{i in J}: psi[i]   = psi[i+1]   - tf/(n-1)/6*(f9[i] + f9[i+1] + 4*f9m[i]);
    
    dynamicdp{i in J}: p[i]    = p[i+1]   - tf/(n-1)/6*(f10[i] + f10[i+1] + 4*f10m[i]);
    dynamicdq{i in J}: q[i]    = q[i+1]   - tf/(n-1)/6*(f11[i] + f11[i+1] + 4*f11m[i]);
    dynamicdr{i in J}: r[i]    = r[i+1]   - tf/(n-1)/6*(f12[i] + f12[i+1] + 4*f12m[i]);
    
    dynamicomega{i in J, j in vJ}: utau[i, j]  = utau[i+1, j] - tf/(n-1)/6*(f13[i, j] + f13[i+1, j] + 4*f13m[i, j]);
#--------------------------------------------------------------------------

#Constraints------------------------------------------
    #Boundary Conditions
    
    #Initial
    subject to InitialPositionx :  dx[1] = dx0;
    subject to InitialPositiony :  dy[1] = dy0;
    subject to InitialPositionz :  dz[1] = dz0;
    
    subject to InitialVelocityx : vx[1] = vx0;
    subject to InitialVelocityy : vy[1] = vy0;
    subject to InitialVelocityz : vz[1] = vz0;
    
    subject to InitialPitch     :  phi[1]   = phi0;
    subject to InitialRoll      :  theta[1] = theta0;
    subject to InitialYaw       :  psi[1]   = psi0;
    
    subject to InitialPitchRate : p[1]   = p0;
    subject to InitialRollRate  : q[1]   = q0;
    subject to InitialYawRate   : r[1]   = r0;
    
    subject to InitialRotorControls{i in vJ} : utau[1,i] = utau0[i];

    #Final
    subject to FinalPositionx :  dx[n] = dxn;
    subject to FinalPositiony :  dy[n] = dyn;
    subject to FinalPositionz :  dz[n] = dzn;
    
    #World coordinates
    #subject to FinalVelocityx : vx[n]*cos(psi[n])*cos(theta[n]) + vy[n]*(sin(phi[n])*sin(theta[n])*cos(psi[n]) - sin(psi[n])*cos(phi[n])) + vz[n]*(sin(phi[n])*sin(psi[n]) + sin(theta[n])*cos(phi[n])*cos(psi[n])) = v_final*cos(psin);
    #subject to FinalVelocityy : vx[n]*sin(psi[n])*cos(theta[n]) + vy[n]*(sin(phi[n])*sin(psi[n])*sin(theta[n]) + cos(phi[n])*cos(psi[n])) + vz[n]*(-sin(phi[n])*cos(psi[n]) + sin(psi[n])*sin(theta[n])*cos(phi[n])) = v_final*sin(psin);
    #subject to FinalVelocityz : -vx[n]*sin(theta[n]) + vy[n]*sin(phi[n])*cos(theta[n]) + vz[n]*cos(phi[n])*cos(theta[n]) = vzn;
    
    subject to FinalVelocityx : vx[n] = vxn;
    subject to FinalVelocityy : vy[n] = vyn;
    subject to FinalVelocityz : vz[n] = vzn;
    
    subject to FinalRoll      : phi[n]   = phin;
    subject to FinalPitch     : theta[n] = thetan;
    subject to FinalYaw       : psi[n] = psin;
    
    subject to FinalRollRate  : p[n]   = pn;
    subject to FinalPitchRate : q[n]   = qn;
    subject to FinalYawRate   : r[n]   = rn;
    
    #subject to FinalRotorControls{i in vJ} : utau[n,i] = u[n,i];
    
    subject to FinalAccelerationZ : f6[n] = 0;
    
    subject to FinalAngularAccelerationP : f10[n] = 0;
    subject to FinalAngularAccelerationQ : f11[n] = 0;
    subject to FinalAngularAccelerationR : f12[n] = 0;
    
    #var x_world{i in I} = -dx[i]*cos(psi[i])*cos(theta[i]) + dy[i]*(-sin(phi[i])*sin(theta[i])*cos(psi[i]) + sin(psi[i])*cos(phi[i])) + dz[i]*(-sin(phi[i])*sin(psi[i]) - sin(theta[i])*cos(phi[i])*cos(psi[i]));
    #var y_world{i in I} = -dx[i]*sin(psi[i])*cos(theta[i]) + dy[i]*(-sin(phi[i])*sin(psi[i])*sin(theta[i]) - cos(phi[i])*cos(psi[i])) + dz[i]*(sin(phi[i])*cos(psi[i]) - sin(psi[i])*sin(theta[i])*cos(phi[i]));
    #var z_world{i in I} = dx[i]*sin(theta[i]) - dy[i]*sin(phi[i])*cos(theta[i]) - dz[i]*cos(phi[i])*cos(theta[i]);
    
    #var dist_to_point{i in I} = (x_world[i])^2 + (y_world[i]-3)^2 + (z_world[i])^2;
    
    #subject to Midpoint : min{i in I} dist_to_point[i] = max_distance;

#-------------------------------------------------------------

#Guess-------------------------------------------------------
    let tf := tn;
    let {i in I, j in vJ}  u[i, j] := 0.5;
#-------------------------------------------------------------

#Solver Options-----------------------------------------------
    option solver snopt;
    option substout 0;
    option show_stats 1;
    options snopt_options "outlev=2 Major_iterations=1500 Superbasics=500";
#-------------------------------------------------------------
