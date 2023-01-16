library(tidyverse)
library(nflfastR)
library(arrow)

path <- "pbp_data"
  

season <- 2022

for (season in 1999:2022) {
  data <- read_parquet(paste(path, paste("play_by_play_",season,".parquet", sep=""), sep="/"))
  data_if_incmp <- data %>%
    mutate(
      failed_fourth_down = ifelse(down == 4, 1, 0),
      ydstogo = ifelse(down < 4, ydstogo, 10),
      posteam = ifelse(down < 4, posteam, defteam),
      down = ifelse(down < 4, down + 1, 1)
    ) %>%
    select(game_id, play_id, season, home_team, posteam, roof, half_seconds_remaining, yardline_100, down, ydstogo, posteam_timeouts_remaining, defteam_timeouts_remaining, failed_fourth_down) %>% 
    nflfastR::calculate_expected_points() %>% 
    mutate(ep_if_incmp=ifelse(failed_fourth_down,-1*ep, ep)) %>% 
    select(game_id, play_id, ep_if_incmp)
  data <- data %>% 
    inner_join(data_if_incmp, by=c("game_id", "play_id")) %>% 
    mutate(epa_if_incmp=ep_if_incmp-ep)
  write_parquet(data, paste(path,"if_incmp",paste("play_by_play_",season,".parquet", sep=""), sep="/"))
}