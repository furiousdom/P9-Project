<template>
  <v-app id="inspire">
    <drawer />
    <v-main>
      <catalog />
    </v-main>
  </v-app>
</template>

<script>
import Catalog from './components';
import Drawer from './components/drawer';

export default {
  name: 'app',
  data: () => ({ drawer: null }),
  components: {
    Catalog,
    Drawer
  }
};

</script>
