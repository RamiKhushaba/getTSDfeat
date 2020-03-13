% -------------------------------------------------------------------
% getTSDfeat get Khushaba et al. Temporal-Spatial Descriptors (TSDs)
% -------------------------------------------------------------------

%% feat = getksmfeat(x,winsize,wininc,datawin,dispstatus)
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

% References
% [1] R. N. Khushaba, A. H. Al-Timemy, A. Al-Ani and A. Al-Jumaily, "A Framework of Temporal-Spatial Descriptors-Based Feature Extraction for Improved Myoelectric Pattern Recognition," in IEEE Transactions on Neural Systems and Rehabilitation Engineering, vol. 25, no. 10, pp. 1821-1831, Oct. 2017. doi: 10.1109/TNSRE.2017.2687520
% [2] A. Al-Timemy, R. N. Khushaba, G. Bugmann, and J. Escudero, "Improving the Performance Against Force Variation of EMG Controlled Multifunctional Upper-Limb Prostheses for Transradial Amputees",
%     IEEE Transactions on Neural Systems and Rehabilitation Engineering, DOI: 10.1109/TNSRE.2015.2445634, 2015.
% [3] R. N. Khushaba, Maen Takruri, Jaime Valls Miro, and Sarath Kodagoda, "Towards limb position invariant myoelectric pattern recognition using time-dependent spectral features",
%     Neural Networks, vol. 55, pp. 42-58, 2014.

function feat = getTSDfeat(x,winsize,wininc,datawin,dispstatus)

if nargin < 5
    if nargin < 4
        if nargin < 3
            if nargin < 2
                winsize = size(x,1);
            end
            wininc = winsize;
        end
        datawin = ones(winsize,1);
    end
    dispstatus = 0;
end

datasize = size(x,1);
Nsignals = size(x,2);
numwin = floor((datasize - winsize)/wininc)+1;

if dispstatus
    h = waitbar(0,'Computing KSM features...');
end
%% prepare indices of each 2 channels combinations
% NCC = Number of channels to combine
NCC  = 2;
Indx = combnk(1:Nsignals,NCC);

%% allocate memory
% define the number of features per channel
NFPC = 7;
% Preallocate memory
feat = zeros(numwin,size(Indx,1)*NFPC+Nsignals*NFPC);
%% prepare windowing analysis parameters
st = 1;
en = winsize;
for i = 1:numwin
    
    curwin = x(st:en,:).*repmat(datawin,1,Nsignals);
    
    %% Step1: Extract between-channels features
    ebp                            = KSM1((curwin(:,Indx(:,1))-curwin(:,Indx(:,2))));
    efp                            = KSM1(log((curwin(:,Indx(:,1))-curwin(:,Indx(:,2))).^2+eps).^2);
    %                              Step2: Correlation analysis
    num                            = ebp.*efp;
    den                            = sqrt(efp.*efp)+sqrt(ebp.*ebp);
    feat(i,1:size(Indx,1)*NFPC)    = num./den;
    
    %% Step1: Extract within-channels features
    ebp                            = KSM1(curwin);
    efp                            = KSM1(log((curwin).^2+eps).^2);
    %                              Step2: Correlation analysis
    num                            = ebp.*efp;
    den                            = sqrt(efp.*efp)+sqrt(ebp.*ebp);
    feat(i,((size(Indx,1)*NFPC)+1):end) = num./den;
    
    %% progress to next window
    st  = st + wininc;
    en  = en + wininc;
end

if dispstatus
    close(h)
end



function Feat = KSM1(S)
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


%% Get the size of the input signal
[samples,channels]=size(S);


%% Root squared zero order moment normalized
m0     = sqrt(sum(S.^2));
m0     = m0.^.1/.1;

% Prepare derivatives for higher order moments
d1     = diff([zeros(1,channels);S],1,1);
d2     = diff([zeros(1,channels);d1],1,1);

% Root squared 2nd and 4th order moments normalized
m2     = sqrt(sum(d1.^2))./(samples-1);
m2     = m2.^.1/.1;

m4     = sqrt(sum(d2.^2))./(samples-1);
m4     = m4.^.1/.1;

%% Sparseness
sparsi = (sqrt(abs((m0-m2).^2.*(m0-m4).^2)).\m0);%sum(d1).^2./sum(d1.^2)-sum(S).^2./sum(S.^2).*sum(d2).^2./sum(d2.^2);

%% Irregularity Factor
IRF    = m2./sqrt(m0.*m4);

%% Coefficient of Variation
COV    = nanstd(S)./nanmean(S);%(sum(abs(d1).*abs(d2)));

%% Teager–Kaiser energy operator
TEA    = nansum(d1.^2 - S(1:end,:).*d2);

%% All features together
STDD = std(m0);
Feat   = (channels>2) * log(abs([(m0)./STDD (m0-m2)./STDD (m0-m4)./STDD sparsi IRF COV TEA])) ...
    + (channels<=2) * log(abs([m0 m0-m2 m0-m4 sparsi IRF COV TEA]));
