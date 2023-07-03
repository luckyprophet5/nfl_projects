library(tidyverse)
library(arrow)

path <- "pbp_data/if_incmp/" # need to run epa_if_incomplete.R first

data <- read_parquet(paste(path, "play_by_play_1999.parquet", sep="/"))
for (season in 2000:2022) {
  data <- rbind(data, read_parquet(paste(path, paste("play_by_play_",season,".parquet", sep=""), sep="/")))
}

dropbacks <- data %>% filter(pass==1)

dropbacks <- dropbacks %>%   
  mutate(
    scramble_epa = ifelse(qb_scramble==1, qb_epa, 0),
    sack_epa = ifelse(sack==1, qb_epa, 0),
    sack_and_scramble_epa = scramble_epa + sack_epa,
    completion_epa = ifelse(complete_pass==1, qb_epa, 0),
    incompletion_epa = ifelse(incomplete_pass==1, qb_epa, 0),
    int_epa = ifelse(interception==1, qb_epa, 0),
    yacoe_epa = ifelse((complete_pass==1) & (fumble_lost!=1) & (!is.na(yac_epa)), yac_epa - replace_na(xyac_epa,0), 0),
    penalty_epa = ifelse((qb_scramble==0)&(sack==0)&(incomplete_pass==0)&(complete_pass==0)&(interception==0)&(penalty==1), qb_epa, 0),
    throw_selection_epa = (air_epa + replace_na(xyac_epa,0))*cp + (epa_if_incmp)*(1-cp),
    accuracy_epa = ifelse(!is.na(cp), completion_epa+incompletion_epa-yacoe_epa-throw_selection_epa, 0)
  )

dropbacks <- dropbacks %>%   
  mutate(
    throw_frequency_epa = ifelse(!is.na(cp), mean(dropbacks$throw_selection_epa, na.rm=TRUE), 0),
    throw_value_epa = throw_selection_epa - throw_frequency_epa
  )


#https://stackoverflow.com/questions/30385626/how-to-get-the-mode-of-a-group-in-summarize-in-r
find_mode <- function(codes){
  which.max(tabulate(codes))
}

seasons <- dropbacks%>% 
  filter(season_type=='REG') %>% 
  group_by(season, passer_id, name) %>% 
  summarize(
    sack_value=sum(sack_epa, na.rm = TRUE)/n(),
    scramble_value=sum(scramble_epa, na.rm = TRUE)/n(),
    int_value=sum(int_epa, na.rm = TRUE)/n(),
    throw_frequency_value=sum(throw_frequency_epa, na.rm = TRUE)/n(),
    throw_value=sum(throw_value_epa, na.rm = TRUE)/n(),
    throw_selection_value=sum(throw_selection_epa, na.rm = TRUE)/n(),
    accuracy_value=sum(accuracy_epa, na.rm = TRUE)/n(),
    yacoe_value=sum(yacoe_epa, na.rm = TRUE)/n(),
    penalty_value=sum(penalty_epa, na.rm = TRUE)/n(),
    epa=sum(qb_epa)/n(),
    dropbacks=n(),
  )


seasons %>% 
  filter((season==2022)&(dropbacks>=272)) %>% 
  arrange(desc(epa)) %>% 
  select(name,sack_value,scramble_value,int_value,throw_frequency_value,throw_value,accuracy_value,yacoe_value,penalty_value,epa,dropbacks)
