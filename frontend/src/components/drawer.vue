<template>
  <v-navigation-drawer
    v-model="drawer"
    permanent
    app
    color="grey lighten-2"
    width="320">
    <v-list>
      <v-list-item>
        <v-list-item-content>
          <v-list-item-title class="title">
            CS-IT9 Project
          </v-list-item-title>
        </v-list-item-content>
      </v-list-item>
      <v-divider />
    </v-list>
  </v-navigation-drawer>
</template>

<script>
export default {
  name: 'drawer'
};
</script>

<style lang="scss" scoped>
.title {
  margin: 1rem 0 1rem 0;
}
</style>
