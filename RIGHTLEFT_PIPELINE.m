%% this is the pipeline to generate a predictive model

tic
%% step 0: load paths and toolboxes

% change to current directory
currentDoc = matlab.desktop.editor.getActive; cd(fileparts(currentDoc.Filename));

TOOLBOXPATH = ['..' filesep '..' filesep 'toolboxes'];
PLOTSPATH = ['..' filesep '..' filesep 'data'];
BACKPATH = ['..'];

if ismac
    % Code to run on Mac plaform
    PATH_PREFIX = ['..' filesep '..' filesep '..' filesep 'Experimental%20Data'];
elseif isunix
    % Code to run on Linux plaform
    PATH_PREFIX = (['..' filesep '..' filesep '..' filesep '..' filesep '..' filesep 'mnt/experimental data/']);
elseif ispc
    % Code to run on Windows platform
    PATH_PREFIX = ['X:'];
    
else
    disp('platform not supported')
end


RAW_DATA_PATH = [PATH_PREFIX filesep '2018-06 RIGHTLEFT'];
PROCESSED_DATA_PATH = [PATH_PREFIX filesep '2018-06 RIGHTLEFT (Processed Data)'];
% set paths to relevant toolboxes
addpath(fullfile(TOOLBOXPATH, 'fieldtrip-20190705'))
%addpath(fullfile(TOOLBOXPATH, 'fieldtrip-20181211', 'external', 'neurone'))
addpath(fullfile(TOOLBOXPATH, 'eeglab13_6_5b')) % needs to have the neurone plug-in installed

[ALLEEG EEG CURRENTSET ALLCOM] = eeglab; % launch eeglab
ft_defaults % initialize fieldtrip


%% step 1: load in data from NeurOne & Unity; high pass filter; epoch
%!!! place your data files accordingly into pipelineEXAMPLE.xlsx...

T = readtable(fullfile('pipelineEXAMPLE.xlsx'), 'Basic', 1);
T = T(strcmpi(T.condition, 'task'), :);
assert(~isempty(T), 'no rows in data table match condition for this condition')

subjects = unique(T.subject)';

subject = {'RIGHTLEFT_02_HeBr'}
%for subject = subjects %!!! if you want to run multiple subjects at once
T_subj = T(strcmpi(T.subject, subject{:}), :);

assert(~isempty(T_subj), 'no rows in data table match condition for this subject')

concatenated_data = [];

%for rowIndex = 1:height(T_subj) %!!! this is if you have multiple sessions
rowIndex = 1
row = T_subj(rowIndex,:);

%!!! ensure that your paths are set correctly
session_filename = fullfile(RAW_DATA_PATH, 'NeurOne Data', row.subject{:}, ['NeurOne-' row.sessionFile{:} '.ses']);
%if in current folder use session_filename = fullfile(['NeurOne-' row.sessionFile{:} '.ses']);

%!!!can comment out if you do not have unity files...
unity_filename = fullfile(RAW_DATA_PATH, 'unity_data', row.subject{:}, [row.unityFile{:} '.txt']);
%if in current folder use unity_filename = fullfile([row.unityFile{:} '.txt']);

%!!!use attached script to investigate (RIGHTLEFT_import_unity)
[hand_used, start_time, hand_out, contact_time, cube_pos] = RIGHTLEFT_import_unity(unity_filename);
handOutTime = hand_out - start_time; %this is in seconds

%!!! pop_readneurone.m calls readneurone.m, which looks up channel location using pop_chanedit, which brings up a dialog
%!!! override with attached script pop_chanedit to return chans without looking up locations to prevent this
EEG = pop_readneurone(session_filename, row.sessionIndex);

marker_time = [EEG.event.latency]/5000;
marker_time = marker_time';
marker_time = marker_time - marker_time(1);

duplicate_marker_indx = find(diff(marker_time) < 2.4) + 1;
fprintf('removing %i duplicate events\n', length(duplicate_marker_indx)); %remove duplicate markers

% remove duplicate events
marker_time(duplicate_marker_indx) = [];
EEG.event(duplicate_marker_indx) = [];

assert(length(EEG.event) == length(hand_used), 'number of entries in unity data does not match number of markers after auto removing duplicates');
start_time = start_time - start_time(1);
hand_out = hand_out - hand_out(1);
assert(max(abs(marker_time - start_time)) < 0.1, 'EEG markers and unity times not aligned (tolerance of 100ms exceeded)');

% high pass filter

%!!! use
EEG = pop_eegfiltnew(EEG,1,0,15000);
%different format; new filter, 10x faster!
%EEG = pop_eegfilt(EEG, 1, 0,5000); %older method

% epoch
%!!!ensure your trigger events correspond here
EEG = pop_epoch(EEG, {'32'}, [-2.5 2.5]); %change interval accordingly

data = eeglab2fieldtrip(EEG, 'preprocessing');

%unity variables
hand_used = [hand_used cube_pos];
data.trialinfo = hand_used;
contact_info = [hand_out contact_time];
data.responseinfo = contact_info;
data.handOutTime = handOutTime;
if isempty(concatenated_data)
    concatenated_data = data;
else
    concatenated_data.trial = {concatenated_data.trial{:} data.trial{:}};
    concatenated_data.time = {concatenated_data.time{:} data.time{:}};
    concatenated_data.trialinfo = [concatenated_data.trialinfo; data.trialinfo];
    concatenated_data.responseinfo = [concatenated_data.responseinfo; data.responseinfo];
    concatenated_data.handOutTime = [concatenated_data.handOutTime; data.handOutTime];
end

%!!! inner for loop (sessions for each subject, if  more than 1 session)
% end )

%!!! save data, uncomment
%save([PROCESSED_DATA_PATH filesep row.subject{:} '_data_pipeline1.mat'], 'concatenated_data', '-v7.3')

%!!! outer loop when used for multiple subjects
%end

%% step 2: filter, downsample, save EMG seperately

responseinfo = concatenated_data.responseinfo;
handOutTime = concatenated_data.handOutTime;
trialinfo = concatenated_data.trialinfo;

% low pass filter data

cfg = [];
cfg.lpfilter   = 'yes';
cfg.lpfreq     = 45; %LOWPASS, highpass already done
%cfg.lpfiltord = 500;
cfg.lpfilttype = 'but';
concatenated_data = ft_preprocessing(cfg, concatenated_data);

% downsample
cfg = [];
cfg.resamplefs = 1000;
concatenated_data = ft_resampledata(cfg, concatenated_data);

% transpose and re-add variables

concatenated_data.responseinfo = responseinfo;
concatenated_data.handOutTime = handOutTime;
concatenated_data.trialinfo =trialinfo;

% save EMG channels seperately
cfg = [];
cfg.channel = {'EMGl', 'EMGr'};
emg = ft_selectdata(cfg, concatenated_data);

% take out EMG
cfg = [];
cfg.channel     = {'all', '-EMGl', '-EMGr'}
concatenated_data = ft_selectdata(cfg, concatenated_data);

% re-save variable to work with ;)
data = concatenated_data;

%save([PROCESSED_DATA_PATH filesep subject '_HP_LP_SAMP1000.mat'], 'data', 'emg', '-v7.3')


% end % inner for loop (sessions for each subject)

%% step 3: do light cleaning, seperate pre & post

myTrials = data.trial;
%want channels by time by trials
b= size (myTrials{1});
a = zeros(b(1), b(2), length(data.trial(:)));

for i = 1:length(a(1,1,:))
    a (:,:,i) = myTrials{i} ;
end

allTrials = a;
timeAxis = data.time{1};
accepted = ones (1,length(data.trial(:)));

%Computing the evoked response with average reference
EEG_ave = mean(allTrials(1:126,:,find(accepted)),3);
EEG_ave_aver= EEG_ave - repmat(mean(EEG_ave,1),[size(EEG_ave,1),1]);

% Visualizing the evoked response
figure(1);
plotData = zeros(126,size(EEG_ave,2));
plotData2=plotData;
plotData(1:126,:) = EEG_ave;
plotData2(1:126,:)=EEG_ave_aver;
plot(timeAxis, plotData);

% take only the PRE movement data ( -2500 to 0ms ) for cleaning
data_pre = data;
%take only the first 2500 time points of each trial for data_pre
trials = data.trial(:);
for i = 1:length(trials)
    tmp = trials {i};
    trials{i} = tmp (:,1:2500);
end
data_pre.trial = trials';

%take only [-2.5 0] for data_pre
time = data.time(:);
for i = 1:length(time)
    tmp = time {i};
    time{i} = tmp (:,1:2500);
end
data_pre.time = time';

% post
data_post = data;
%take only the last 2500 time points of each trial for data_pre
trials = data.trial(:);
for i = 1:length(trials)
    tmp = trials {i};
    trials{i} = tmp (:,2501:4999);
end
data_post.trial = trials';

%take only [0 2.5] for datapost
time = data.time(:);
for i = 1:length(time)
    tmp = time {i};
    time{i} = tmp (:,2501:4999);
end
data_post.time = time';

% reject contaminated channels and trials
%!!!copy these electrodes to delete for default peripheral cleaning;
% F9 F10 FFT9h FFT10h FT9 FT10 FT7 FT8 FTT8h FFT7h T7 T8 TP7 TP8 TTP7h TTP8h TP10 TP9 TPP10h TPP9h P8 P7 P9 P10 PPO10h PPO9h PO9 PO10 POO10h POO9h O9 O10 OI1h OI2h Iz Oz O1 O2 POO1 POO2 PO7 PO8
reject_info = [];
reject_info.chans  = {}; % keep record of rejected channels
reject_info.trials = []; % keep record of rejected trials

data_pre.sampleinfo = ones(length(data_pre.trial),2);
for i = 1:size(data_pre.sampleinfo,1)
    data_pre.sampleinfo(i,1) = data_pre.sampleinfo(i,1)+1000*(i-1);
    data_pre.sampleinfo(i,2) = data_pre.sampleinfo(i,1)-1+1000;
end

%create layout
cfg = [];
cfg.layout = 'EEG1005.lay';
layout = ft_prepare_layout(cfg);

cfg = [];
cfg.layout = layout;
cfg.channel = {'all'};
cfg.preproc.demean = 'yes';

cfg.method = 'summary'
cfg.artfctdef.reject = 'complete';
cfg.artfctdef.feedback = 'yes';

[data_visualrejected] = ft_rejectvisual(cfg, data_pre);

reject_info.chans           = setxor(data_pre.label,data_visualrejected.label);
reject_info.trials = {} ; %manual input...copyandpaste...todo: FIX THIS

% check out data_pre

myTrials = data_visualrejected.trial;
%want channels by time by trials
b= size (myTrials{1});
a = zeros(b(1), b(2), length(data_visualrejected.trial(:)));

for i = 1:length(a(1,1,:))
    a (:,:,i) = myTrials{i} ;
end

allTrials = a;
timeAxis = data_visualrejected.time{1};
accepted = ones (1,length(data_visualrejected.trial(:)));

%Computing the evoked response with average reference
EEG_ave = mean(allTrials(1:length(data_visualrejected.label),:,find(accepted)),3);
EEG_ave_aver= EEG_ave - repmat(mean(EEG_ave,1),[size(EEG_ave,1),1]);

% Visualizing the evoked response
figure(1);
plotData = zeros(length(data_visualrejected.label),size(EEG_ave,2));
plotData2=plotData;
plotData(1:length(data_visualrejected.label),:) = EEG_ave;
plotData2(1:length(data_visualrejected.label),:)=EEG_ave_aver;
plot(timeAxis, plotData);

% now do channel-wise cleaning
% reject contaminated channels and trials

data_pre.sampleinfo = ones(length(data_pre.trial),2);
for i = 1:size(data_pre.sampleinfo,1)
    data_pre.sampleinfo(i,1) = data_pre.sampleinfo(i,1)+1000*(i-1);
    data_pre.sampleinfo(i,2) = data_pre.sampleinfo(i,1)-1+1000;
end

%create layout
cfg = [];
cfg.layout = 'EEG1005.lay';
layout = ft_prepare_layout(cfg);

cfg = [];
cfg.layout = layout;
cfg.channel = {'all'};
cfg.preproc.demean = 'yes';

cfg.method = 'channel'
cfg.artfctdef.reject = 'complete';
cfg.artfctdef.feedback = 'yes';

[data_visualrejected] = ft_rejectvisual(cfg, data_visualrejected);


reject_info2.chans           = setxor(data_pre.label,data_visualrejected.label);
reject_info2.trials = {} ;
%manual input...copyandpaste...todo: FIX THIS
%channels sum up in reject infos, but the trials are independently kept
%i.e. info+info2.chans = info2.chans while info+info2.trials = info+info2.trials

% rereference to the mean
cfg = [];
cfg.reref          = 'yes';
cfg.refchannel     = 'CPz'; % {'all', '-EMGl', '-EMGr'}
%cfg.implicitref    = 'FCz';
cfg.refmethod      = 'avg';

REREFdata = ft_preprocessing(cfg, data_visualrejected);

% check out data_pre again

myTrials = REREFdata.trial;
%want channels by time by trials
b= size (myTrials{1});
a = zeros(b(1), b(2), length(REREFdata.trial(:)));

for i = 1:length(a(1,1,:))
    a (:,:,i) = myTrials{i} ;
end

allTrials = a;
timeAxis = REREFdata.time{1};
accepted = ones (1,length(REREFdata.trial(:)));

%Computing the evoked response with average reference
EEG_ave = mean(allTrials(1:length(REREFdata.label),:,find(accepted)),3);
EEG_ave_aver= EEG_ave - repmat(mean(EEG_ave,1),[size(EEG_ave,1),1]);

% Visualizing the evoked response
figure(1);
plotData = zeros(length(REREFdata.label),size(EEG_ave,2));
plotData2=plotData;
plotData(1:length(REREFdata.label),:) = EEG_ave;
plotData2(1:length(REREFdata.label),:)=EEG_ave_aver;
plot(timeAxis, plotData);

% BASELINE CORRECTION
cfg.demean         = 'yes';
cfg.baselinewindow = [-2 -1];
preICA_data = ft_preprocessing(cfg, REREFdata);

% check out data_pre again

myTrials = preICA_data.trial;
%want channels by time by trials
b= size (myTrials{1});
a = zeros(b(1), b(2), length(preICA_data.trial(:)));

for i = 1:length(a(1,1,:))
    a (:,:,i) = myTrials{i} ;
end

allTrials = a;
timeAxis = preICA_data.time{1};
accepted = ones (1,length(preICA_data.trial(:)));

%Computing the evoked response with average reference
EEG_ave = mean(allTrials(1:length(preICA_data.label),:,find(accepted)),3);
EEG_ave_aver= EEG_ave - repmat(mean(EEG_ave,1),[size(EEG_ave,1),1]);

% Visualizing the evoked response
figure(1);
plotData = zeros(length(preICA_data.label),size(EEG_ave,2));
plotData2=plotData;
plotData(1:length(preICA_data.label),:) = EEG_ave;
plotData2(1:length(preICA_data.label),:)=EEG_ave_aver;
plot(timeAxis, plotData);

%% step 4: run ICA to remove eye blinks
% run ICA
%create data for visualization
cfg = [];
cfg.channels={'all'};
cfg.resamplefs = 1000;
cfg.method = 'fastica';
cfg.fastica.approach = 'symm';
cfg.fastica.g = 'gauss';
cfg.fastica.numOfIC = 50;
ICA_comp1 = ft_componentanalysis(cfg, preICA_data);

% --> 1st ROUND of ICA
% Visualize topoplot and time course (averaged) to decide which components to reject
% SUPER DEBORA PLOTS for ICA
nfig = ceil(length(ICA_comp1.label)./4);

topoICA = [];
topoICA.time = 1;
topoICA.label = ICA_comp1.topolabel;
topoICA.fsample = 1000;
topoICA.dimord = 'chan_time';

cfg =[];
cfg.layout = layout;
cfg.parameter = 'topography';
cfg.comment = 'no';

comp = 0;
for ff = 1 :nfig
    figure
    pp = 0;
    while pp< 4 && comp<length(ICA_comp1.label)
        pp = pp+1;
        comp = comp+1;
        topoICA.topography = ICA_comp1.topo(:, comp);
        
        subplot(4, 4, 1+4*(pp-1))
        ft_topoplotER(cfg,topoICA);
        colormap jet
        title(['ICA comp'  num2str(comp)], 'FontSize', 18)
        
        timecourse = cell2mat(cellfun(@(x) x(comp,:),ICA_comp1.trial,'UniformOutput',false)');
        timecourse_pre= timecourse(:,1:490);
        fourierspctrm = fft(timecourse_pre')';
        fourierspctrm = fourierspctrm(:,1:size(timecourse_pre, 2)/2+1);
        powspctrm = (1/(topoICA.fsample*size(timecourse_pre, 2)))*abs(fourierspctrm).^2;
        powspctrm(:,2:end-1) = 2*powspctrm(:,2:end-1);
        freq =  0:topoICA.fsample/size(timecourse_pre, 2):topoICA.fsample/2;
        subplot(4, 4, 2+4*(pp-1))
        plot(ICA_comp1.time{1},mean(timecourse,1),'LineWidth',1,'Color',[64 64 64]./255)
        % xlim([-0.300 0.500])
        ylim([-150 150])
        title(['Average timecourse ICA comp'  num2str(comp)])
        
        subplot(4, 4, 3+4*(pp-1))
        imagesc(ICA_comp1.time{1},1:size(timecourse,1),timecourse, [-5*mean(mean(abs(timecourse))) 5*mean(mean(abs(timecourse)))])
        colormap jet
        title(['Single trial timecourse ICA comp'  num2str(comp)])
        
        subplot(4, 4, 4+4*(pp-1))
        plot(freq(freq>=4 & freq<= 100),mean(powspctrm(:,freq>=4 & freq<= 100)),'LineWidth',1,'Color',[64 64 64]./255)
        title(['Power spectrum ICA comp'  num2str(comp)])
        set(gcf, 'Units', 'Normalized', 'OuterPosition', [0 0 1 1]);
    end
end

% save figs, and save components for further work
%saveFiguresAsPDF;
% save([PROCESSED_DATA_PATH filesep subject '_ICAcomp.mat'], 'ICA_comp1', 'preICA_data', 'data', 'data_post', 'reject_info2', 'emg','-v7.3');




%% ICA: Reject components and save data
%(before running this section, modify the componets to reject!)
%REJECT THE COMPONENTS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
reject_comp1= [2 29 44 ]; %<--HERE ENTER BAD COMPONENTS TO REJECT
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

close all
cfg.component = reject_comp1;
cfg.updatesens   = 'yes';
[posICA_data] = ft_rejectcomponent(cfg,ICA_comp1, preICA_data);
ICA_comp1.excluded=reject_comp1;
%save(proces_data, 'ICA_comp1','-append')

%%% pre-processing data after ICA 1st round



%%% Prepare data for second round ICA
cfg = [];
cfg.method = 'fastica';
cfg.fastica.approach = 'symm';
cfg.fastica.g = 'gauss';
cfg.fastica.numOfIC = 32;

ICA_comp2 = ft_componentanalysis(cfg, posICA_data);
%ICA_comp2 are the components in component space
%% CHECK data for eyeblinks * Final Clean
cfg = [];
cfg.viewmode = 'vertical';
cfg.channel = {'AFp1','AFp2'};
ft_databrowser(cfg, posICA_data);
%!! it is OK if there are no channels to display
%
%save ([CLEANED_DATA_PATH filesep subject 'predata_ICA.mat'], 'ICA_comp1','ICA_comp2', 'posICA_data','data','data_post' ,'reject_info','reject_info2',  'emg','-v7.3');

%% step 5: put back in the post-stimulus data, rerefernece it, baseline it
% take out indx from reject_info and apply to response times
trial_removed_data = data_post;
tmp = [];

for i = 1:length(reject_info.trials)
    tmp (i) = reject_info.trials{i};
end
trial_removed_data.responseinfo(tmp,:) = [];
trial_removed_data.trialinfo(tmp,:) = [];
trial_removed_data.trial(tmp) = [];
trial_removed_data.time(tmp) = [];
trial_removed_data.handOutTime(tmp) = [];
additionalVariables.trialinfo_org = data.trialinfo;

% repeat for reject_info2 (because trials numbers shifted!)
tmp = [];

for i = 1:length(reject_info2.trials)
    tmp (i) = reject_info2.trials{i};
end
trial_removed_data.responseinfo(tmp,:) = [];
trial_removed_data.trialinfo(tmp,:) = [];
trial_removed_data.trial(tmp) = [];
trial_removed_data.time(tmp) = [];
trial_removed_data.sampleinfo = posICA_data.sampleinfo;
trial_removed_data.handOutTime(tmp) = [];
response_times = trial_removed_data.responseinfo(:,1); %double check name here
additionalVariables.handOutTime = trial_removed_data.handOutTime;
additionalVariables.responseinfo = trial_removed_data.responseinfo;
additionalVariables.response_times = response_times;


% need to take out channels from "trials"
%this returns sharedvals which are the channels to remove,
% and at the positions we want to remove (idx)...
[sharedvals,idx] = intersect(trial_removed_data.label,reject_info2.chans,'stable'); %check reject_info(X)

%so now... remove these idx in the labels of chans...
trial_removed_data.label(idx) = [];

%and now, we need to take out these index from the cells of the trials
%because, normally these are '128' by 2500, and they need to be length('label') by 2500

tmp = [];
for i = 1:length(trial_removed_data.trial)
    tmp = trial_removed_data.trial{i};
    tmp (idx,:) = [];
    trial_removed_data.trial{i} = tmp;
end

% now, the post data has the right channels and trials removed, but we need to add the weights!!! matrix multiplication
%something like, trial_removed_data.trial * ICA_comp2...
%ICA_comp2.trial is 32 x 2500
%ICA_comp2.unmixing is 32 x 104...
%trial_removed_data.trial is 104 x 2499

%trial_removed_data.trial{1}' * ICA_comp2.unmixing'
data_post_ICA_weighted = cellfun(@(x) ICA_comp2.topo * (ICA_comp2.unmixing * x), trial_removed_data.trial, 'UniformOutput', false);

trial_removed_data.trial = data_post_ICA_weighted;
trial_removed_data.elec = posICA_data.elec;
trial_removed_data.sampleinfo = posICA_data.sampleinfo;
trial_removed_data.cfg = posICA_data.cfg;

% rereference to the mean, do not add the implicit reference channel to the data
cfg = [];
cfg.reref          = 'yes';
cfg.refchannel     = 'CPz'; % {'all', '-EMGl', '-EMGr'}
%cfg.implicitref    = 'FCz';
cfg.refmethod      = 'avg';
cfg.baselinewindow = [2 2.5];
cfg.demean         = 'yes';
data_post_reref = ft_preprocessing(cfg, trial_removed_data);
data_pre_reref = posICA_data;



% clean up EMG data now
% take out indx from reject_info and apply to response times
tmp = [];

for i = 1:length(reject_info.trials)
    tmp (i) = reject_info.trials{i};
end
emg.responseinfo(tmp,:) = [];
emg.trialinfo(tmp,:) = [];
emg.trial(tmp) = [];
emg.time(tmp) = [];
emg.handOutTime(tmp) = [];

% repeat for reject_info2 (because trials numbers shifted!)
tmp = [];

for i = 1:length(reject_info2.trials)
    tmp (i) = reject_info2.trials{i};
end
emg.responseinfo(tmp,:) = [];
emg.trialinfo(tmp,:) = [];
emg.trial(tmp) = [];
emg.time(tmp) = [];
emg.sampleinfo = posICA_data.sampleinfo;
emg.handOutTime(tmp) = [];


% now let's append the trials from data_post...!
data_total = data_pre_reref;
tmp = [];
for i = 1:length(data_total.trial)
    tmp = data_total.trial {i};
    tmp2 = data_post_reref.trial {i};
    data_total.trial{i} = [tmp tmp2];
end

tmp = [];
for i = 1:length(data_total.time)
    tmp = data_total.time {i};
    tmp2 = data_post_reref.time {i};
    data_total.time{i} = [tmp tmp2];
end

%% save pre and post data structures with all relevant infomation
% save ([CLEANED_DATA_PATH filesep subject '_handCleaned_ALLdata_August_postICA_reref.mat'],'ICA_comp2', 'data_post_reref','data_pre_reref','data_total','reject_info','reject_info2','emg', 'additionalVariables','-v7.3');

%% step 6: use data total to generate model variables


subNum = 0;
%subjects = [];

for subject = subjects

%subject = {'RIGHTLEFT_02_HeBr'}
%subject = string(subject);
subNum = subNum + 1;
data_total = data_working{subNum}

trialInfo = data_total.trialinfo(:,1);

% settings for PIPELINE
trials_all   = find(trialInfo)
%trials_left = find(trialInfo == 1)
%trials_right =find(trialInfo == 2)
inds= trials_all;
ts=data_total.time{1};


%!!! ENTER YOUR TIME WINDOW HERE, FOR US, 150 is START OF MOVEMENT
t1=  -150;% time interval start
t2=  150;% time interval end
t1 = t1/1000;
t2 = t2/1000;
[~, it1]=min(abs(ts-(t1)));
[~, it2]=min(abs(ts-(t2)));

npc = 30;

k = 1; %cross-validations

%% needed variables
Xtotal=cat(3,data_total.trial{:});
y_total=trialInfo;

Xsub=Xtotal(:,it1:it2,inds);
y=y_total(inds);
[C, T, R]=size(Xsub);


cvFolds = crossvalind('Kfold', y, k);   % get indices of k-fold
cp = classperf(y);
cp2 = classperf(y);

% for each fold
trainIdx = (cvFolds == 1);                % get indices of test instances
testIdx = trainIdx;  %edit accordingly here                   % get indices training instances


Xsubsub=Xsub(:,:,trainIdx);
meanXsub=mean(Xsub(:,:,trainIdx),3);
xd=Xsub-meanXsub; %mean-subtraction if wanted %!!
xtilde=reshape(Xsub, C,[]);

xcov2=reshape(Xsubsub, C,[])*reshape(Xsubsub, C,[])'/R*T;
[u, d,~ ]=svds(xcov2, npc); %pca

%Pcomp=diag(diag(d(1:npc, 1:npc)).^(-.5))*u(:,1:npc)';
Pcomp=u(:,1:npc)';
XtildeComp=Pcomp*reshape(xd(:,:,trainIdx), C,[]);
[~, Atilde, Wtilde]=fastica([XtildeComp], 'g', 'tanh', 'approach', 'symm', 'verbose', 'off');
%[~, ~, Wtilde]=fastica([XtildeComp], 'g', 'tanh', 'approach', 'symm', 'verbose', 'off');
% features
Sd=Wtilde*Pcomp*reshape(xd, C, []);
Sd=reshape(Sd,size(Sd,1),T,R);
S=Wtilde*Pcomp*xtilde;
S=reshape(S,size(S,1),T,R);
%features
S1=squeeze(mean(S,2)); %mean
S2=(squeeze(mean(Sd.^2,2))); %variance


% train an SVM model over training instances
features=[S1;S2]'; %all features

X= features(trainIdx,:); %training sets
Y = y(trainIdx);


% rng default
Mdl = fitclinear(X,Y,'Solver','sparsa','Learner','Logistic',...
    'OptimizeHyperparameters', {'Lambda'},'HyperparameterOptimizationOptions',...
    struct('AcquisitionFunctionName','expected-improvement-plus','MaxObjectiveEvaluations',100,'ShowPlots',false,'Verbose',0));

% test using test instances
Z = features(testIdx,:);
pred = predict(Mdl,Z);
% evaluate and update performance object
cp = classperf(cp, pred, testIdx);

var1=cp.ErrorRate;
disp(['error rate',num2str(var1)]);

%% save variables
MODELVARS{subNum} ={var2, Atilde, Wtilde, Mdl, Pcomp, xcov2, xd, xtilde};
end
%% save
MODELVARS_readme = '{var2, Atilde, Wtilde, Mdl, TF, Pcomp, xcov2, xd, xtilde}';
%  save('FINALVAR_vs_PIPELINE.mat', 'FINALVAR_vs_PIPELINE','FINALVAR_readme','-v7.3')
toc
%% DONE


