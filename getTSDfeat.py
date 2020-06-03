"""
% -------------------------------------------------------------------
% getTSDfeat get Khushaba et al. Temporal-Spatial Descriptors (TSDs)
% -------------------------------------------------------------------

%% feat = getTSDfeat(x,winsize,wininc)
%
% Author Rami Khushaba 
%
% This function computes the temporal-spatial descriptors of the signals in x,
% x is made of columns, each representing a channel/sensor.
% For example if you get 5 sec of data from 8 channels/sensors at 1000 Hz
% then x should be 5000 x 8. A windowing scheme is used here to extract features
%
% The signals in x are divided into multiple windows of size winsize and the windows are slid by wininc.
%
% Inputs
%    x: 		columns of signals
%    winsize:	window size (length of x)
%    wininc:	spacing of the windows (winsize)
%    datawin:   window for data (e.g. Hamming, default rectangular)
%               must have dimensions of (winsize,1)
%    dispstatus:zero for no waitbar (default)
%
% Outputs
%    feat:     TSDs (6 features per channel/channels' difference)
%
% Modifications
% 23/06/2004   AC: template created http://www.sce.carleton.ca/faculty/chan/index.php?page=matlab
% 17/11/2013   RK: Spectral moments first created.
% 01/03/2014   AT: Rami Sent me this on 1-3-14 to go with normalised KSM_V1
% 01/02/2016   RK: Modifed this code into somewhat deep structure
% 03/06/2020   RK: ported the function to Python.

% References
% [1] R. N. Khushaba, A. H. Al-Timemy, A. Al-Ani and A. Al-Jumaily, "A Framework of Temporal-Spatial Descriptors-Based Feature Extraction for Improved Myoelectric Pattern Recognition," in IEEE Transactions on Neural Systems and Rehabilitation Engineering, vol. 25, no. 10, pp. 1821-1831, Oct. 2017. doi: 10.1109/TNSRE.2017.2687520
% [2] A. Al-Timemy, R. N. Khushaba, G. Bugmann, and J. Escudero, "Improving the Performance Against Force Variation of EMG Controlled Multifunctional Upper-Limb Prostheses for Transradial Amputees",
%     IEEE Transactions on Neural Systems and Rehabilitation Engineering, DOI: 10.1109/TNSRE.2015.2445634, 2015.
% [3] R. N. Khushaba, Maen Takruri, Jaime Valls Miro, and Sarath Kodagoda, "Towards limb position invariant myoelectric pattern recognition using time-dependent spectral features",
%     Neural Networks, vol. 55, pp. 42-58, 2014.
"""
import numpy as np
from itertools import combinations 

def getTSDfeat(x,*slidingParams):
    
    # x should be a numpy array
    x = np.array(x)
    
    # Make sure you have the correct number of parameters passed
    if len(slidingParams) <2:
        raise TypeError('getTSDfeat expected winsize and wininc to be passed, got %d parameters instead' %len(slidingParams))
    if slidingParams:
        winsize = slidingParams[0]
        wininc = slidingParams[1]
    
    if len(x.shape)==1:
        x=x[:,np.newaxis]
        
    datasize = x.shape[0]
    Nsignals = x.shape[1]
    numwin = np.int(np.floor((datasize - winsize)/wininc)+1)
    
    # prepare indices of each 2 channels combinations
    # NCC = Number of channels to combine
    NCC  = 2
    Indx = np.array(list(combinations(range(Nsignals), NCC)))

    
    # allocate memory
    # define the number of features per channel
    NFPC = 7;
    
    # Preallocate memory
    feat = np.zeros((numwin,Indx.shape[0]*NFPC+Nsignals*NFPC))
    
    # prepare windowing analysis parameters
    st = 0
    en = winsize
    
    for i in range(numwin):
        
        # define your current window
        curwin = x[st:en,:]
    
        # Step1.1: Extract between-channels features
        ebp                                 = KSM1(curwin[:,Indx[:,0]]-curwin[:,Indx[:,1]])
        efp                                 = KSM1(np.log((curwin[:,Indx[:,0]]-curwin[:,Indx[:,1]])**2+np.spacing(1))**2)
        # Step 1.2: Correlation analysis
        num                                 = np.multiply(efp,ebp)
        den                                 = np.sqrt(np.multiply(efp,efp))+np.sqrt(np.multiply(ebp,ebp))
        feat[i,range(Indx.shape[0]*NFPC)]   = num/den
        
        # Step2.1: Extract within-channels features
        ebp                            = KSM1(curwin)
        efp                            = KSM1(np.log((curwin)**2+np.spacing(1))**2)
        # Step2.2: Correlation analysis
        num                            = np.multiply(efp,ebp)
        den                            = np.sqrt(np.multiply(efp,efp))+np.sqrt(np.multiply(ebp,ebp))
        feat[i,np.max(range(Indx.shape[0]*NFPC))+1:] = num/den
        
        # progress to next window
        st  = st + wininc
        en  = en + wininc
    
    return feat
        
def KSM1(S):
    """
    % Time-domain power spectral moments (TD-PSD)
    % Using Fourier relations between time domina and frequency domain to
    % extract power spectral moments dircetly from time domain.
    %
    % Modifications
    % 17/11/2013  RK: Spectral moments first created.
    % 02/03/2014  AT: I added 1 to the function name to differentiate it from other versions from Rami
    % 01/02/2016  RK: Modifed this code intosomewhat deep structure
    %
    % References
    % [1] A. Al-Timemy, R. N. Khushaba, G. Bugmann, and J. Escudero, "Improving the Performance Against Force Variation of EMG Controlled Multifunctional Upper-Limb Prostheses for Transradial Amputees",
    %     IEEE Transactions on Neural Systems and Rehabilitation Engineering, DOI: 10.1109/TNSRE.2015.2445634, 2015.
    % [2] R. N. Khushaba, Maen Takruri, Jaime Valls Miro, and Sarath Kodagoda, "Towards limb position invariant myoelectric pattern recognition using time-dependent spectral features",
    %     Neural Networks, vol. 55, pp. 42-58, 2014.
    """
    
    # Get the size of the input signal
    samples,channels = S.shape
    
    if channels>samples:
        S  = np.transpose(S)
        samples,channels = channels, samples
        
    # Root squared zero order moment normalized
    m0     = np.sqrt(np.nansum(S**2,axis=0))[:,np.newaxis]
    m0     = m0**.1/.1
    
    # Prepare derivatives for higher order moments
    d1     = np.diff(np.concatenate([np.zeros((1,channels)),S],axis=0),n=1,axis=0)
    d2     = np.diff(np.concatenate([np.zeros((1,channels)),d1],axis=0),n=1,axis=0)

    # Root squared 2nd and 4th order moments normalized
    m2     = (np.sqrt(np.nansum(d1**2,axis=0))/(samples-1))[:,np.newaxis]
    m2     = m2**.1/.1
    
    m4     = (np.sqrt(np.nansum(d2**2,axis=0))/(samples-1))[:,np.newaxis]
    m4     = m4**.1/.1
    
    # Sparseness
    sparsi = m0/np.sqrt(np.abs(np.multiply((m0-m2)**2,(m0-m4)**2)))
    
    # Irregularity Factor
    IRF    = m2/np.sqrt(np.multiply(m0,m4))
    
    # Coefficient of Variation
    COV    = (np.nanstd(S,axis=0, ddof=1)/np.nanmean(S,axis=0))[:,np.newaxis]
    
    # Teager-Kaiser energy operator
    TEA    = np.nansum(d1**2 - np.multiply(S[0:samples,:],d2),axis=0)[:,np.newaxis]
    
    # All features together
    STDD = np.nanstd(m0,axis=0, ddof=1)[:,np.newaxis]
    
    if channels>2:
        Feat   = np.concatenate((m0/STDD, (m0-m2)/STDD, (m0-m4)/STDD,sparsi, IRF, COV, TEA), axis=0)
    else:
        Feat   = np.concatenate((m0, m0-m2, m0-m4,sparsi, IRF, COV, TEA), axis=0)
     
    Feat   = np.log(np.abs(Feat)).flatten()
    
    return Feat  